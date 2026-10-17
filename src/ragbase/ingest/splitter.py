"""文档切分层：负责生成适合检索的片段并做片段级去重。"""
import re
import unicodedata
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


@dataclass
class SplittingReport:
    input_documents: int = 0
    raw_chunks: int = 0
    output_chunks: int = 0
    empty_chunks_removed: int = 0
    duplicate_chunks_removed: int = 0


def create_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """创建中文友好的递归字符切分器。"""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )


def normalized_chunk_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", "", text)


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[list[Document], SplittingReport]:
    """递归切分并删除文件内部的空片段和完全重复片段。"""
    report = SplittingReport(input_documents=len(documents))
    chunks = create_splitter(chunk_size, chunk_overlap).split_documents(documents)
    report.raw_chunks = len(chunks)

    unique_chunks = []
    seen = set()
    for chunk in chunks:
        key = normalized_chunk_key(chunk.page_content)
        if not key:
            report.empty_chunks_removed += 1
            continue
        if key in seen:
            report.duplicate_chunks_removed += 1
            continue
        seen.add(key)
        unique_chunks.append(chunk)

    report.output_chunks = len(unique_chunks)
    return unique_chunks, report
