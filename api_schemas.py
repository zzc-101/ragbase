"""API 的请求 / 响应模型。"""
import os
from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

Mode = Literal["vector", "bm25", "hybrid"]


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
    scores: Scores


class Citation(BaseModel):
    n: int = Field(description="引用编号，对应回答里的 [片段n]")
    chunk_id: str
    text: str
    source: str = Field(description="来源文件名（不含目录）")
    page: int | None = Field(None, description="页码（1 起，仅 PDF 有）")
    scores: Scores


class SearchRequest(BaseModel):
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


class AskRequest(BaseModel):
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


class HealthResponse(BaseModel):
    status: Literal["ok", "not_ready"]
    chunk_count: int | None = Field(description="索引片段数，索引不存在为 null")
    bm25_ready: bool
    device: str
    mode: Mode
    llm_configured: bool


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
