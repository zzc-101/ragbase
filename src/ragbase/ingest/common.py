"""RAG 公共配置、稳定 ID 与入库编排工具。"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from ragbase.config import get_settings, load_env
from ragbase.ingest.cleaner import clean_documents
from ragbase.ingest.parser import load_document
from ragbase.ingest.splitter import split_documents
from ragbase.ingest.toc import pdf_page_headings

load_env()  # 载入项目主目录（RAG_HOME，默认当前目录）下的 .env

COLLECTION = "rag"
INGEST_PIPELINE_VERSION = "3"  # 解析/清洗/切分/章节标题规则变化时递增（3：PDF 片段加 heading_path）


def file_fingerprint(path: Path) -> str:
    """计算文件指纹：路径 + 大小 + 修改时间。

    用于判断文件是否有变化（增删改检测）。
    不用内容哈希是因为大文件读一遍太慢，mtime+size 足够检测改动。
    """
    stat = path.stat()
    raw = (
        f"{INGEST_PIPELINE_VERSION}|{path.name}|"
        f"{stat.st_size}|{int(stat.st_mtime)}"
    )
    return hashlib.md5(raw.encode()).hexdigest()


def file_id(path: Path) -> str:
    """每个文件的稳定 ID（基于文件名，不含路径中的特殊字符）。"""
    return hashlib.md5(path.name.encode()).hexdigest()[:16]


def chunk_id(file_idx: str, chunk_index: int) -> str:
    """单个片段的 ID：文件ID + 片段序号，保证同一文件的片段可整批删除/重建。"""
    return f"{file_idx}_{chunk_index}"


def build_ingest_meta(paths: list[Path]) -> dict:
    """生成当前文档集合的增量索引元数据。"""
    return {
        file_id(path): {"name": path.name, "fp": file_fingerprint(path)}
        for path in paths
    }


def load_ingest_meta(meta_path: Path | None = None) -> dict:
    """读取上次成功入库时保存的文件元数据（默认路径取配置里的 meta_file，记录文件指纹，用于增量更新）。"""
    meta_path = meta_path or get_settings().meta_file
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def save_ingest_meta(meta: dict, meta_path: Path | None = None) -> None:
    """原子写入文件元数据，避免中断时留下半截 JSON。"""
    meta_path = meta_path or get_settings().meta_file
    temp_file = meta_path.with_suffix(".tmp")
    temp_file.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_file.replace(meta_path)


def with_heading(text: str, metadata: dict, separator: str) -> str:
    """片段有章节路径（heading_path）时，把它拼在正文前面；没有则原样返回。"""
    heading = (metadata or {}).get("heading_path")
    return f"{heading}{separator}{text}" if heading else text


def embedding_input(text: str, metadata: dict) -> str:
    """向量化的输入：章节路径 + 换行 + 正文（Chroma 里存的 document 仍是纯正文）。"""
    return with_heading(text, metadata, "\n")


def bm25_input(text: str, metadata: dict) -> str:
    """BM25 分词的输入：章节路径 + 空格 + 正文。"""
    return with_heading(text, metadata, " ")


def collection_count(db_dir: str | Path | None = None, name: str = COLLECTION) -> int | None:
    """返回集合片段数；索引目录或集合不存在时返回 None（不加载任何模型）。db_dir 缺省取配置里的 db_dir。"""
    db_dir = db_dir or get_settings().db_dir
    if not Path(db_dir).exists():  # PersistentClient 会自动建目录，所以先判断
        return None
    import chromadb

    try:
        return chromadb.PersistentClient(path=str(db_dir)).get_collection(name).count()
    except Exception:
        return None


def prepare_file_chunks(path: Path):
    """编排解析、清洗和切分，返回 Chroma 载荷及预处理报告。"""
    fid = file_id(path)
    parsed_documents = load_document(path)
    # PDF 书签 -> 每页章节路径；其他格式暂无章节信息
    page_headings = pdf_page_headings(path) if path.suffix.lower() == ".pdf" else []
    cleaned_documents, cleaning_report = clean_documents(parsed_documents)
    chunks, splitting_report = split_documents(cleaned_documents)
    texts, metadatas, ids = [], [], []
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)
        metadata["fid"] = fid
        metadata["chunk_index"] = index
        metadata["content_hash"] = hashlib.sha256(
            chunk.page_content.encode("utf-8")
        ).hexdigest()[:16]
        page = metadata.get("page")
        if isinstance(page, int) and 0 <= page < len(page_headings) and page_headings[page]:
            metadata["heading_path"] = page_headings[page]  # 未知时不写该字段（Chroma 元数据只能是标量）
        texts.append(chunk.page_content)
        metadatas.append(metadata)
        ids.append(chunk_id(fid, index))
    report = {
        "parsed_documents": len(parsed_documents),
        "cleaning": asdict(cleaning_report),
        "splitting": asdict(splitting_report),
        "heading_chunks": sum(1 for metadata in metadatas if metadata.get("heading_path")),
    }
    return texts, metadatas, ids, report
