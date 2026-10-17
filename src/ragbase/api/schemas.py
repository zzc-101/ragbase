"""API 的请求 / 响应模型。"""
import os
from typing import Any, Literal

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

Mode = Literal["vector", "bm25", "hybrid"]
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"  # 会话 ID：只接受安全字符（会进日志 / URL）


class Scores(BaseModel):
    vec_rank: int | None = Field(None, description="向量路内排名（1 起），未命中为 null")
    bm25_rank: int | None = Field(None, description="BM25 路内排名（1 起），未命中为 null")
    rrf_score: float | None = Field(None, description="RRF 融合分数")
    rerank_score: float | None = Field(None, description="精排得分 P(相关)，0~1；精排前的候选为 null")


class Chunk(BaseModel):
    rank: int = Field(description="在所在列表中的位置，1 起")
    chunk_id: str
    text: str
    source: str = Field(description="来源文件名（不含目录）")
    page: int | None = Field(None, description="页码（1 起，仅 PDF 有）")
    heading_path: str | None = Field(None, description="章节路径（PDF 书签），如 第11章 > 11.3 挣值分析；旧索引 / 无书签为 null")
    scores: Scores


class Citation(BaseModel):
    n: int = Field(description="引用编号，对应回答里的 [片段n]")
    chunk_id: str
    text: str
    source: str = Field(description="来源文件名（不含目录）")
    page: int | None = Field(None, description="页码（1 起，仅 PDF 有）")
    heading_path: str | None = Field(None, description="章节路径（PDF 书签）；旧索引 / 无书签为 null")
    scores: Scores


class SourcesField(BaseModel):
    """/search 与 /ask 共用的按书过滤字段。"""

    sources: list[str] | None = Field(
        None,
        max_length=50,
        description="只在这些书里检索（文件名，取自 GET /documents 的 name）；不传或空列表表示全部；有未知书名返回 400",
    )


class SearchRequest(SourcesField):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=1000, description="问题（首尾空白会被去掉，不能为空）")
    mode: Mode | None = Field(None, description="召回方式，默认取服务配置")
    top_k: int | None = Field(None, ge=1, le=50, description="精排后返回的片段数，默认取 RAG_RERANK_K")
    min_score: float | None = Field(None, ge=0, le=1, description="拒答阈值，0 表示不拒答；默认取服务配置")
    include_candidates: bool = Field(False, description="是否同时返回精排前的候选片段")


class SearchResponse(BaseModel):
    refused: bool = Field(description="最相关片段得分低于 min_score（或无召回）；仍会返回 chunks 供查看")
    top_score: float | None
    chunks: list[Chunk]
    candidates: list[Chunk] | None = None
    took_ms: float


class AskRequest(SourcesField):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=1000, description="问题（首尾空白会被去掉，不能为空）")
    mode: Mode | None = Field(None, description="召回方式，默认取服务配置")
    min_score: float | None = Field(None, ge=0, le=1, description="拒答阈值，0 表示不拒答；默认取服务配置")
    context_min_score: float | None = Field(
        None, ge=0, le=1, description="只把精排得分不低于该值的片段交给大模型，0 表示不过滤；默认取服务配置",
    )


class AskResponse(BaseModel):
    answer: str
    evidence: str = Field(description="大模型逐字摘录的依据，没有则为空串")
    refused: bool = Field(description="是否因得分过低 / 无召回拒答（未调用大模型）")
    top_score: float | None
    citations: list[Citation] = Field(description="citations[n-1] 对应回答里的 [片段n]；拒答时为空")
    took_ms: float


class ChatRequest(SourcesField):
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str | None = Field(
        None, pattern=SESSION_ID_PATTERN, description="会话 ID（字母 / 数字 / _ / -，最长 64）；不传则新建并在响应里返回"
    )
    message: str = Field(min_length=1, max_length=2000, description="用户消息（首尾空白会被去掉，不能为空）")


class ToolCallInfo(BaseModel):
    name: str = Field(description="工具名：search_knowledge / list_documents")
    args: dict[str, Any] = Field(description="模型传给工具的参数")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[Citation] = Field(description="回答里实际标注了 [n] 的片段；n 与回答中的 [n] 一致（可能不连续），编号在整个会话内稳定，回答沿用历史里的编号时也能解析")
    tool_calls: list[ToolCallInfo] = Field(description="本轮模型调用过的工具（按顺序）")
    took_ms: float


class SessionMessage(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str = Field(description="工具消息只给前 80 字的摘要")
    tool_calls: list[ToolCallInfo] | None = Field(None, description="助手消息发起的工具调用")
    citations: list[Citation] = Field(default_factory=list, description="助手消息正文里标注了 [n] 的片段（会话级编号）；其他角色为空")


class SessionResponse(BaseModel):
    session_id: str
    messages: list[SessionMessage]


class SessionSummary(BaseModel):
    session_id: str
    title: str = Field(description="第一条用户消息（单行，最多 30 字，超出加省略号）")
    updated_at: str = Field(description="最近使用时间（UTC，ISO 8601）")
    message_count: int = Field(description="用户消息与最终回答的条数（不含工具调用步骤）")


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary] = Field(description="按最近使用倒序；Agent 未就绪时为空列表")


class SessionDeleteResponse(BaseModel):
    session_id: str
    deleted: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "not_ready"]
    chunk_count: int | None = Field(description="索引片段数，索引不存在为 null")
    bm25_ready: bool
    bm25_built_at: str | None = Field(None, description="当前 BM25 索引的构建时间（UTC，ISO 8601）；未构建为 null")
    bm25_chunk_count: int | None = Field(None, description="当前 BM25 索引覆盖的片段数；与 chunk_count 不同说明刷新尚未完成")
    device: str
    mode: Mode
    llm_configured: bool
    vector_store: str = Field(description="向量库后端（RAG_VECTOR_STORE）：chroma / qdrant")
    llm_protocol: str = Field(description="大模型协议（RAG_LLM_PROTOCOL），目前只有 openai")
    llm_model: str | None = Field(description="回答模型名（RAG_LLM_MODEL）；未配置为 null")
    agent_ready: bool = Field(description="对话 Agent（/chat）是否就绪；大模型未配置或索引缺失时为 false")
    auth_required: bool = Field(False, description="是否启用了接口鉴权（设置了 RAG_API_KEY）")


class LivenessResponse(BaseModel):
    """启用鉴权后，没有有效密钥的 /health 只返回这两个字段（存活探针 + 让前端知道需要密钥）。"""

    status: Literal["ok", "not_ready"]
    auth_required: Literal[True] = True


class DocumentInfo(BaseModel):
    name: str = Field(description="文件名（不含目录），可用于 sources 过滤和 DELETE /documents/{name}")
    chunk_count: int
    heading_chunks: int = Field(description="带章节路径的片段数")
    has_headings: bool = Field(description="是否有章节路径（旧索引或无书签的 PDF 为 false）")


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]


class JobResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "succeeded", "failed"]
    files_total: int = 0
    files_done: int = 0
    current_file: str | None = None
    chunks_added: int = 0
    chunks_deleted: int = 0
    skipped_files: list[str] = Field(default_factory=list, description="被跳过的文件（旧版 .doc）")
    error: str | None = None
    started_at: str | None = Field(None, description="UTC，ISO 8601")
    finished_at: str | None = None


class UploadResponse(JobResponse):
    filename: str = Field(description="实际保存的文件名（已去掉目录部分）")


class DeleteResponse(BaseModel):
    name: str
    chunks_deleted: int


class ErrorResponse(BaseModel):
    error: str
    detail: str


def _chunk_fields(doc: Document) -> dict:
    metadata = doc.metadata
    page = metadata.get("page")
    return {
        "chunk_id": str(metadata.get("chunk_id", "")),
        "text": doc.page_content,
        "source": os.path.basename(metadata.get("source", "?")),
        # 元数据里的 page 是 0 起，对外与命令行 / 评测一致，用 1 起的页码
        "page": page + 1 if isinstance(page, int) else None,
        "heading_path": metadata.get("heading_path") or None,
        "scores": Scores(
            vec_rank=metadata.get("vec_rank"),
            bm25_rank=metadata.get("bm25_rank"),
            rrf_score=metadata.get("rrf_score"),
            rerank_score=metadata.get("rerank_score"),
        ),
    }


def to_chunk(rank: int, doc: Document) -> Chunk:
    return Chunk(rank=rank, **_chunk_fields(doc))


def to_citation(n: int, doc: Document) -> Citation:
    return Citation(n=n, **_chunk_fields(doc))
