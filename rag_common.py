"""RAG 公共配置与文档入库工具。"""
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = str(BASE_DIR / "chroma_db")
COLLECTION = "rag"
META_FILE = BASE_DIR / ".ingest_meta.json"  # 记录文件指纹，用于增量更新

# 支持的文件格式 -> 后缀集合
SUPPORTED_EXTS = {".pdf", ".docx", ".doc", ".md", ".markdown", ".txt"}

# 切分器（中文友好：按段落 -> 句号 -> 逗号）
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
)


def load_document(path: Path):
    """按文件格式选择合适的 loader，返回 Document 列表。

    支持格式：PDF / Word(.docx) / Markdown / 纯文本
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(str(path))
    elif ext in (".docx", ".doc"):
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(str(path))
    elif ext in (".md", ".markdown", ".txt"):
        from langchain_community.document_loaders import TextLoader
        # text_loader_kwargs 解决部分文件编码问题
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    else:
        raise ValueError(f"不支持的格式：{ext}（仅支持 {SUPPORTED_EXTS}）")
    return loader.load()


def list_documents() -> list[Path]:
    """列出 data/ 下所有支持的文档文件。"""
    files = []
    for ext in SUPPORTED_EXTS:
        files.extend(DATA_DIR.glob(f"*{ext}"))
    return sorted(files)


def file_fingerprint(path: Path) -> str:
    """计算文件指纹：路径 + 大小 + 修改时间。

    用于判断文件是否有变化（增删改检测）。
    不用内容哈希是因为大文件读一遍太慢，mtime+size 足够检测改动。
    """
    stat = path.stat()
    raw = f"{path.name}|{stat.st_size}|{int(stat.st_mtime)}"
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


def prepare_file_chunks(path: Path):
    """解析并切分单个文件，返回可直接写入 Chroma 的文本、元数据和稳定 ID。"""
    fid = file_id(path)
    chunks = SPLITTER.split_documents(load_document(path))
    texts, metadatas, ids = [], [], []
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)
        metadata["fid"] = fid
        texts.append(chunk.page_content)
        metadatas.append(metadata)
        ids.append(chunk_id(fid, index))
    return texts, metadatas, ids
