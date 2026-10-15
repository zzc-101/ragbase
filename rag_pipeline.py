"""查询与评测共用的 RAG 流水线。"""
import os
import re
from dataclasses import dataclass

import chromadb
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from embed import Qwen3Embeddings
from hybrid_retriever import HybridRetriever
from rag_common import COLLECTION, DB_DIR
from reranker import Qwen3Reranker


NO_ANSWER = "我没有在文档中找到相关内容。"  # 召回为空或精排得分过低时的固定回答

# 最相关片段的精排得分（P(相关)，0~1）低于该值就拒答、不调用 LLM。
# 校准（scripts/calibrate_rerank_threshold.py）：45 道库内问题 top1 得分 0.937~0.9999（中位 0.9994），
# 21 道库外问题 0.0000~0.9882（14 道 < 0.06，其余 0.61~0.99 与库内有重叠）。
# 取最低库内分 × 0.75 ≈ 0.7：库内零误拒，拦下 15/21；宁可放过也不误拒。
RERANK_MIN_SCORE = 0.7

# 通过拒答门槛后，只把 rerank_score >= 该值的片段交给 LLM（最相关的 top1 总是保留）；None/0 不过滤。
# 数据（临时实验脚本，30 道评测题、hybrid、精排 Top5、不调用 LLM）：Reranker 得分高度饱和，
# 库内题全部 5 个片段都 >= 0.96（Top5 最低 0.9612），所以 <= 0.95 的阈值对库内题等于不过滤：
#   阈值    平均保留片段   要点召回(point_recall)
#   0~0.95     5.00          0.8344（与不过滤相同）
#   0.97       4.87          0.8210（-0.013，超出 0.01 容忍）
#   0.99       4.63          0.7988
#   0.998      3.37          0.7121
# 取满足“要点召回不低于未过滤值 0.01”的最高候选 0.95：库内题零损失，只在片段得分明显偏低时才裁剪。
# 注意得分已饱和，提高阈值会开始裁掉对比 / 多方面题需要的片段，别盲目调高。
CONTEXT_MIN_SCORE = 0.95

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


@dataclass
class RAGPipeline:
    retriever: object
    reranker: Qwen3Reranker
    answer_chain: object
    embeddings: Qwen3Embeddings


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
    retrieve_k: int = 20,
    mode: str = "hybrid",
    per_route_k: int = 30,
) -> tuple[RAGPipeline, ChatOpenAI]:
    """从本地缓存加载模型，并构建检索、精排和生成组件。

    mode: vector（仅向量）/ bm25（仅关键词）/ hybrid（两路 RRF 融合）。
    """
    # Transformers 并行初始化两个同系列 Qwen 模型时存在偶发 meta tensor 竞态，
    # 因此这里顺序加载；local_files_only 快路径已把冷启动压缩到数秒。
    embeddings = Qwen3Embeddings()
    reranker = Qwen3Reranker()
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION)
    retriever = HybridRetriever(
        collection,
        embeddings,
        mode=mode,
        per_route_k=per_route_k,
        final_k=retrieve_k,
    )

    llm = ChatOpenAI(
        model=os.getenv("ARK_CHAT_MODEL"),
        api_key=os.getenv("ARK_CHAT_API_KEY"),
        base_url=os.getenv("ARK_BASE_URL"),
        temperature=0,
    )
    return RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        answer_chain=build_answer_chain(llm),
        embeddings=embeddings,
    ), llm


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[片段{i + 1}] {doc.page_content}"
        for i, doc in enumerate(docs)
    )


def retrieve_and_rerank(pipeline: RAGPipeline, question: str, rerank_k: int = 5):
    """召回并精排（不调用 LLM），返回 (精排片段, 原始候选片段)。

    精排片段是候选的副本，metadata 里附带 rerank_score（P(相关)，0~1）；
    原始候选保持不变。
    """
    candidates = pipeline.retriever.invoke(question)
    if not candidates:
        return [], []  # 空召回不调用 reranker
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
) -> AskResult:
    """执行召回、精排和生成，返回 AskResult。

    无召回结果，或最相关片段的 rerank_score < min_score 时直接返回 NO_ANSWER、
    不调用 LLM（reranked 仍会返回，便于查看最接近的片段）。min_score 为 None 或 0 时不按得分拒答。
    通过后只把 rerank_score >= context_min_score 的片段交给 LLM（None/0 不过滤），
    LLM 输出的【依据】/【回答】会被拆开。
    """
    reranked, candidates = retrieve_and_rerank(pipeline, question, rerank_k)
    top_score = reranked[0].metadata["rerank_score"] if reranked else None
    if should_refuse(reranked, min_score):
        return AskResult(NO_ANSWER, "", [], reranked, candidates, True, top_score)
    docs = select_context_docs(reranked, context_min_score)
    raw = pipeline.answer_chain.invoke({
        "context": format_docs(docs),
        "input": question,
    })
    evidence, answer = split_evidence_answer(raw)
    return AskResult(answer, evidence, docs, reranked, candidates, False, top_score)
