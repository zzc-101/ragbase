"""查询与评测共用的 RAG 流水线。"""
import logging
import re
import threading
import time
from dataclasses import dataclass, field

import chromadb
import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import CONTEXT_MIN_SCORE, RERANK_MIN_SCORE, Settings, get_settings, resolve_device
from embed import Qwen3Embeddings
from hybrid_retriever import HybridRetriever
from llm_factory import create_chat_model
from rag_common import COLLECTION
from reranker import Qwen3Reranker


logger = logging.getLogger(__name__)

NO_ANSWER = "我没有在文档中找到相关内容。"  # 召回为空或精排得分过低时的固定回答

# 拒答阈值 RERANK_MIN_SCORE 与上下文过滤阈值 CONTEXT_MIN_SCORE 的取值依据见 config.py
# （从那里 import，本模块照旧可导入）。

ANSWER_SYSTEM_PROMPT = f"""你是一个文档问答助手。请只根据下面提供的上下文片段回答用户的问题。

请先摘录依据，再作答，严格按以下格式输出：
【依据】
- [片段N] “从该片段逐字摘录的原文”
【回答】
基于依据的回答，句末标注 [片段N]

规则：
1. 依据必须是片段原文的逐字摘录，不得改写；每条不超过 80 字，最多 5 条。
2. 回答只能使用【依据】里已有的事实，不要补充片段中没有的事实、数据或常识。
3. 可以综合多条依据整合说明；如果片段只覆盖了问题的一部分，只回答有依据的部分，并明确指出哪一部分在文档中没有找到。
4. 如果所有片段都与问题无关，只回复“{NO_ANSWER}”，不要输出【依据】【回答】等任何其他内容。
5. 回答尽量简洁、准确。

上下文片段：
{{context}}"""

EVIDENCE_HEADING = re.compile(
    r"^[ \t#*]*(?:[【\[(（]\s*依据\s*[】\])）]|依据\s*[:：])[ \t*:：]*",
    re.MULTILINE,
)
ANSWER_HEADING = re.compile(
    r"^[ \t#*]*(?:[【\[(（]\s*回答\s*[】\])）]|回答\s*[:：])[ \t*:：]*",
    re.MULTILINE,
)


class GenerationError(RuntimeError):
    """调用大模型生成回答失败（网络、鉴权、限流等）；原始异常在 __cause__ 里。

    timeout 为 True 表示是响应超时（API 层据此返回 504 而不是 502）。
    """

    def __init__(self, message: str, timeout: bool = False):
        super().__init__(message)
        self.timeout = timeout


def is_timeout_error(exc: BaseException | None) -> bool:
    """沿异常链判断是否超时：内置 TimeoutError、httpx 的超时异常、openai 的 APITimeoutError（按类名，避免依赖具体 SDK）。"""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
            return True
        if any(cls.__name__ == "APITimeoutError" for cls in type(exc).__mro__):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


@dataclass
class RAGPipeline:
    retriever: object
    reranker: Qwen3Reranker
    answer_chain: object  # 未配置大模型时为 None（只能检索，不能 ask）
    embeddings: Qwen3Embeddings
    device: str = ""
    # 本地 embedding / reranker 在 MPS 上不能并发使用：召回 + 精排期间持有该锁（不包含 LLM 调用）
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def build_answer_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    return (
        {"context": lambda x: x["context"], "input": lambda x: x["input"]}
        | prompt
        | llm
        | StrOutputParser()
    )


def build_pipeline(
    retrieve_k: int | None = None,
    mode: str | None = None,
    per_route_k: int | None = None,
    device: str | None = None,
    build_bm25: bool | None = None,
    settings: Settings | None = None,
) -> tuple[RAGPipeline, BaseChatModel | None]:
    """从本地缓存加载模型，并构建检索、精排和生成组件。

    未显式传入的参数取自 settings（默认 get_settings()）：召回数、每路召回数、默认模式、设备、回答模型。
    mode: vector（仅向量）/ bm25（仅关键词）/ hybrid（两路 RRF 融合）。
    build_bm25: 是否构建 BM25 索引，默认 mode != vector 时构建；API 服务要按请求切换模式，传 True。
    回答模型未配置齐全时 answer_chain / llm 为 None，仍可检索。
    """
    settings = settings or get_settings()
    retrieve_k = retrieve_k or settings.retrieve_k
    per_route_k = per_route_k or settings.per_route_k
    mode = mode or settings.mode
    device = resolve_device(device or settings.device)

    # Transformers 并行初始化两个同系列 Qwen 模型时存在偶发 meta tensor 竞态，
    # 因此这里顺序加载；local_files_only 快路径已把冷启动压缩到数秒。
    start = time.perf_counter()
    embeddings = Qwen3Embeddings(device=device)
    reranker = Qwen3Reranker(device=device)
    logger.info("本地模型加载完成（设备 %s），用时 %.1fs", device, time.perf_counter() - start)
    client = chromadb.PersistentClient(path=str(settings.db_dir))
    collection = client.get_collection(COLLECTION)
    retriever = HybridRetriever(
        collection,
        embeddings,
        mode=mode,
        per_route_k=per_route_k,
        final_k=retrieve_k,
        build_bm25=build_bm25,
        cache_dir=settings.cache_dir,
    )

    llm = create_chat_model(settings)  # 协议不支持时在这里（服务启动阶段）抛 ValueError
    if llm is None:
        logger.warning(
            "大模型未配置（RAG_LLM_MODEL / RAG_LLM_API_KEY / RAG_LLM_BASE_URL，旧的 ARK_* 变量同样有效），只能检索不能生成回答"
        )
    return RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        answer_chain=build_answer_chain(llm) if llm is not None else None,
        embeddings=embeddings,
        device=device,
    ), llm


def format_docs(docs) -> str:
    """拼成交给 LLM 的上下文；片段有章节路径时写在编号后面：[片段1]（第11章 > 11.3 挣值分析）正文。"""
    parts = []
    for i, doc in enumerate(docs):
        heading = doc.metadata.get("heading_path")
        label = f"[片段{i + 1}]（{heading}）" if heading else f"[片段{i + 1}] "
        parts.append(f"{label}{doc.page_content}")
    return "\n\n".join(parts)


def retrieve_and_rerank(
    pipeline: RAGPipeline,
    question: str,
    rerank_k: int = 5,
    mode: str | None = None,
    sources: list[str] | None = None,
):
    """召回并精排（不调用 LLM），返回 (精排片段, 原始候选片段)。

    精排片段是候选的副本，metadata 里附带 rerank_score（P(相关)，0~1）；
    原始候选保持不变。mode 为 None 时用检索器构造时的默认模式；
    sources（书名 = 文件 basename 列表）不为空时只在这些书里检索。
    全程持有 pipeline.lock：本地模型不支持并发推理，多线程（API）调用时在此串行。
    """
    options = {}
    if mode is not None:
        options["mode"] = mode
    if sources:
        options["sources"] = sources
    with pipeline.lock:
        candidates = pipeline.retriever.invoke(question, **options)
        if not candidates:
            return [], []  # 空召回不调用 reranker
        # 精排输入只用正文，不拼章节路径：拒答阈值（RERANK_MIN_SCORE）是按纯正文的得分分布校准的，
        # 改了输入会让分数分布漂移、阈值失效。章节路径只参与向量化和 BM25 分词。
        ranked = pipeline.reranker.rerank_with_scores(
            question,
            [doc.page_content for doc in candidates],
            top_k=rerank_k,
        )
    docs = [
        candidates[index].model_copy(update={
            "metadata": {**candidates[index].metadata, "rerank_score": score},
        })
        for index, score in ranked
    ]
    return docs, candidates


def should_refuse(docs, min_score: float | None) -> bool:
    """没有精排片段，或最相关片段得分低于 min_score 时应拒答；min_score 为 None/0 则只看是否为空。"""
    if not docs:
        return True
    if not min_score:
        return False
    return docs[0].metadata.get("rerank_score", 1.0) < min_score


def select_context_docs(docs, context_min_score: float | None):
    """只保留 rerank_score >= context_min_score 的片段（保持精排顺序）；top1 总是保留。

    context_min_score 为 None/0 时不过滤；没有 rerank_score 的片段视为满分保留。
    """
    if not context_min_score:
        return list(docs)
    return [
        doc
        for index, doc in enumerate(docs)
        if index == 0 or doc.metadata.get("rerank_score", 1.0) >= context_min_score
    ]


def _strip_block(text: str) -> str:
    """去掉首尾空白，并把连续空行压成一个。"""
    return re.sub(r"\n\s*\n+", "\n", text.strip())


def split_evidence_answer(text: str) -> tuple[str, str]:
    """把 LLM 输出拆成 (依据, 回答)。

    - 按【依据】/【回答】格式（半角/全角括号、“依据：”“回答：”标题、多余空行均可）-> (依据, 回答)
    - 只输出 NO_ANSWER -> ("", NO_ANSWER)
    - 只有【回答】没有依据 -> ("", 回答)
    - 没按格式输出（含只有依据没有回答）-> ("", 全文)
    """
    text = (text or "").strip()
    if text.strip("。.\"“” ") == NO_ANSWER.rstrip("。"):
        return "", NO_ANSWER
    answer_heading = ANSWER_HEADING.search(text)
    if not answer_heading:
        return "", text
    answer = _strip_block(text[answer_heading.end():])
    if not answer:
        return "", text
    evidence = ""
    evidence_heading = EVIDENCE_HEADING.search(text)
    if evidence_heading and evidence_heading.end() <= answer_heading.start():
        evidence = _strip_block(text[evidence_heading.end():answer_heading.start()])
    return evidence, answer


@dataclass
class AskResult:
    """ask 的返回值。

    answer: 最终回答（已去掉【依据】部分）；evidence: 逐字摘录的依据（无则为空）；
    docs: 实际交给 LLM 的片段，编号与 [片段N] 一一对应（拒答时为空）；
    reranked: 精排后的全部片段（含被过滤掉的，拒答时也有，便于查看最接近的片段）；
    candidates: 精排前的原始候选；refused: 是否因空召回 / 得分过低拒答（未调用 LLM）；
    top_score: 最相关片段的精排得分（无精排结果为 None）。
    """

    answer: str
    evidence: str
    docs: list
    reranked: list
    candidates: list
    refused: bool
    top_score: float | None


def ask(
    pipeline: RAGPipeline,
    question: str,
    rerank_k: int = 5,
    min_score: float | None = RERANK_MIN_SCORE,
    context_min_score: float | None = CONTEXT_MIN_SCORE,
    mode: str | None = None,
    sources: list[str] | None = None,
) -> AskResult:
    """执行召回、精排和生成，返回 AskResult。

    无召回结果，或最相关片段的 rerank_score < min_score 时直接返回 NO_ANSWER、
    不调用 LLM（reranked 仍会返回，便于查看最接近的片段）。min_score 为 None 或 0 时不按得分拒答。
    通过后只把 rerank_score >= context_min_score 的片段交给 LLM（None/0 不过滤），
    LLM 输出的【依据】/【回答】会被拆开。mode 为 None 时用默认召回模式；sources 限定检索的书；
    大模型调用失败抛 GenerationError（不在锁内，慢生成不会阻塞别的检索）。
    """
    reranked, candidates = retrieve_and_rerank(pipeline, question, rerank_k, mode, sources)
    top_score = reranked[0].metadata["rerank_score"] if reranked else None
    if should_refuse(reranked, min_score):
        return AskResult(NO_ANSWER, "", [], reranked, candidates, True, top_score)
    docs = select_context_docs(reranked, context_min_score)
    try:
        raw = pipeline.answer_chain.invoke({
            "context": format_docs(docs),
            "input": question,
        })
    except Exception as exc:
        if is_timeout_error(exc):
            raise GenerationError(f"大模型响应超时：{type(exc).__name__}", timeout=True) from exc
        raise GenerationError(f"大模型调用失败：{type(exc).__name__}") from exc
    evidence, answer = split_evidence_answer(raw)
    return AskResult(answer, evidence, docs, reranked, candidates, False, top_score)
