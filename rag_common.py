"""RAG 公共配置、稳定 ID 与入库编排工具。"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from document_cleaner import clean_documents
from document_parser import load_document
from document_splitter import split_documents

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = str(BASE_DIR / "chroma_db")
COLLECTION = "rag"
META_FILE = BASE_DIR / ".ingest_meta.json"  # 记录文件指纹，用于增量更新
INGEST_PIPELINE_VERSION = "2"  # 解析/清洗/切分规则变化时递增


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


def load_ingest_meta() -> dict:
    """读取上次成功入库时保存的文件元数据。"""
    if not META_FILE.exists():
        return {}
    return json.loads(META_FILE.read_text(encoding="utf-8"))


def save_ingest_meta(meta: dict) -> None:
    """原子写入文件元数据，避免中断时留下半截 JSON。"""
    temp_file = META_FILE.with_suffix(".tmp")
    temp_file.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_file.replace(META_FILE)


def collection_count(db_dir: str = DB_DIR, name: str = COLLECTION) -> int | None:
    """返回集合片段数；索引目录或集合不存在时返回 None（不加载任何模型）。"""
    if not Path(db_dir).exists():  # PersistentClient 会自动建目录，所以先判断
        return None
    import chromadb

    try:
        return chromadb.PersistentClient(path=db_dir).get_collection(name).count()
    except Exception:
        return None


def prepare_file_chunks(path: Path):
    """编排解析、清洗和切分，返回 Chroma 载荷及预处理报告。"""
    fid = file_id(path)
    parsed_documents = load_document(path)
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
        texts.append(chunk.page_content)
        metadatas.append(metadata)
        ids.append(chunk_id(fid, index))
    report = {
        "parsed_documents": len(parsed_documents),
        "cleaning": asdict(cleaning_report),
        "splitting": asdict(splitting_report),
    }
    return texts, metadatas, ids, report
