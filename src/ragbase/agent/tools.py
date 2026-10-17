"""对话 Agent 的工具：search_knowledge（检索知识库）与 list_documents（列出书目），以及会话级的引用登记表。

多轮改写不单独做一步：search_knowledge 的描述要求模型把 query 写成不依赖上下文的完整问句
（代词 / 省略已在对话里补全），检索直接用这个 query。

引用编号：登记表按会话保存（随会话存活、被淘汰），多轮共用——片段编号在整个会话内唯一且稳定
（同一个 chunk_id 永远是同一个编号，后续轮次的新片段接着往下编），最终回答里的 [n] 才能对应到 citations；
模型直接沿用历史里出现过的编号作答时，也能解析到对应片段。某一轮失败回滚时，用 mark / rollback 撤销该轮新登记的片段。
登记表走 LangGraph 的运行时上下文（create_agent 的 context_schema + 工具里的 ToolRuntime）：
每次 agent.invoke / stream 都传入一个 ChatContext（带上该会话的登记表），工具从 runtime.context 取到它——
比 ContextVar 可靠：工具在线程池里执行，ContextVar 需要靠拷贝上下文才能传到，而显式传入的 context 不依赖这一点。
"""
import logging
import os
import re
import threading
from dataclasses import dataclass, field

from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document

from ragbase.config import Settings
from ragbase.retrieval.hybrid import UnknownSourceError
from ragbase.retrieval.pipeline import RAGPipeline, retrieve_and_rerank, select_context_docs, should_refuse

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 1000  # 与 /search 的 query 上限一致
NOT_FOUND_MESSAGE = "知识库中没有找到可靠的相关内容。请直接告诉用户“知识库中没有相关内容”，不要凭自己的知识编造答案。"

SEARCH_DESCRIPTION = """检索知识库（《系统架构设计师》《信息系统项目管理师》等教材）。凡是涉及知识库里教材内容的问题，都必须先调用本工具再回答。

参数 query 必须是一个自包含的完整问题：把对话里的代词和省略补全（例如用户问“那它的计算公式呢？”，要写成“挣值分析中进度绩效指数 SPI 的计算公式是什么？”），不要依赖上下文；一次只问一个方面，多方面的问题可分多次调用。
参数 sources（可选）是书名列表，只在这些书里检索；不确定书名时先调用 list_documents。

返回若干带编号的片段，格式为“[编号] 来源：《书名》> 章节路径（第N页）”加“内容：……”。回答时用 [编号] 标注依据。"""

LIST_DESCRIPTION = "列出知识库里有哪些书（书名和片段数）。用户询问知识库有什么内容，或需要用 search_knowledge 的 sources 参数限定某本书时调用。"


class ToolRetrievalError(RuntimeError):
    """检索本身出错（模型加载、索引损坏等），区别于大模型调用失败；原始异常在 __cause__ 里。"""


class CitationRegistry:
    """会话级的引用登记表：同一 chunk_id 返回同一个编号（1 起，跨轮不重置），线程安全。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._numbers: dict[str, int] = {}
        self._docs: list[Document] = []

    @staticmethod
    def _key(doc: Document) -> str:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id:
            return f"id:{chunk_id}"
        return f"text:{doc.metadata.get('source', '')}:{doc.metadata.get('page', '')}:{hash(doc.page_content)}"

    def add(self, doc: Document) -> int:
        """登记片段并返回编号；已登记过的片段返回原编号（保留首次登记的得分等元数据）。"""
        key = self._key(doc)
        with self._lock:
            number = self._numbers.get(key)
            if number is None:
                self._docs.append(doc)
                number = self._numbers[key] = len(self._docs)
            return number

    def all(self) -> list[tuple[int, Document]]:
        with self._lock:
            return list(enumerate(self._docs, 1))

    def mark(self) -> int:
        """当前已登记的片段数；轮次开始时记下，失败回滚时传给 rollback。"""
        with self._lock:
            return len(self._docs)

    def rollback(self, mark: int) -> None:
        """撤销 mark 之后登记的片段（编号是按登记顺序连续分配的，所以直接截断即可）。"""
        with self._lock:
            del self._docs[mark:]
            self._numbers = {key: number for key, number in self._numbers.items() if number <= mark}

    def clear(self) -> None:
        self.rollback(0)

    def cited_in(self, answer: str) -> list[tuple[int, Document]]:
        """回答文本里实际出现过 [n] 的已登记片段（按编号升序，含历史轮次登记的）；编号保持不变，所以可能不连续。"""
        referenced = {int(match) for match in re.findall(r"\[(\d{1,4})\]", answer or "")}
        return [(number, doc) for number, doc in self.all() if number in referenced]


@dataclass
class ChatContext:
    """每次 agent 调用传入的运行时上下文：所属会话的引用登记表 + 请求里指定的默认书目过滤。"""

    registry: CitationRegistry = field(default_factory=CitationRegistry)
    default_sources: list[str] | None = None


def _book_title(source: str) -> str:
    return os.path.splitext(os.path.basename(source or "?"))[0]


def format_fragment(number: int, doc: Document) -> str:
    """[3] 来源：《书名》> 章节路径（第45页）\\n内容：……；没有章节 / 页码（旧索引、非 PDF）则省略对应部分。"""
    metadata = doc.metadata
    label = f"[{number}] 来源：《{_book_title(metadata.get('source', '?'))}》"
    if metadata.get("heading_path"):
        label += f"> {metadata['heading_path']}"
    page = metadata.get("page")
    if isinstance(page, int):
        label += f"（第{page + 1}页）"  # 元数据里的 page 是 0 起
    return f"{label}\n内容：{doc.page_content}"


def build_tools(pipeline: RAGPipeline, settings: Settings) -> list:
    """用给定的检索流水线构建工具；流水线在服务运行期间不变，所以闭包捕获即可。"""

    @tool("search_knowledge", description=SEARCH_DESCRIPTION)
    def search_knowledge(query: str, runtime: ToolRuntime[ChatContext], sources: list[str] | None = None) -> str:
        query = (query or "").strip()[:MAX_QUERY_CHARS]
        if not query:
            return "query 不能为空：请给出一个完整的问题。"
        context = runtime.context
        names = list(dict.fromkeys(sources or [])) or context.default_sources
        if names:
            valid = [entry["name"] for entry in pipeline.retriever.catalog()]
            unknown = [name for name in names if name not in valid]
            if unknown:  # 不抛异常：把可选项告诉模型，让它自己纠正
                return f"未知的书名 {unknown}。可选的书名：{valid}。请改用其中的书名，或不传 sources 检索全部。"
        try:
            reranked, _ = retrieve_and_rerank(pipeline, query, settings.rerank_k, None, names)
        except UnknownSourceError as exc:  # 书目在校验后被删除的竞态
            return f"未知的书名 {exc.unknown}。可选的书名：{exc.valid}。"
        except Exception as exc:
            logger.exception("知识库检索失败")
            raise ToolRetrievalError("知识库检索失败") from exc
        if should_refuse(reranked, settings.min_score):
            return NOT_FOUND_MESSAGE
        docs = select_context_docs(reranked, settings.context_min_score)
        return "\n\n".join(format_fragment(context.registry.add(doc), doc) for doc in docs)

    @tool("list_documents", description=LIST_DESCRIPTION)
    def list_documents() -> str:
        catalog = pipeline.retriever.catalog()
        if not catalog:
            return "知识库目前没有任何书。"
        lines = [f"- {entry['name']}（{entry['chunk_count']} 个片段）" for entry in catalog]
        return f"知识库共 {len(catalog)} 本书（search_knowledge 的 sources 请使用下面的文件名）：\n" + "\n".join(lines)

    return [search_knowledge, list_documents]
