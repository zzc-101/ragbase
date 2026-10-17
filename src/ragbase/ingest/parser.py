"""文档解析层：负责把不同文件格式转换为 LangChain Document。"""
import threading
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document


SUPPORTED_EXTS = {".pdf", ".docx", ".md", ".markdown", ".txt"}
LEGACY_EXTS = {".doc"}  # 旧版二进制 Word，docx2txt 读不了，只提示不入库


def is_ignored_name(name: str) -> bool:
    """README 说明文件、隐藏文件和 Word 临时锁文件（~$开头）不当作文档。"""
    return Path(name).stem.lower() == "readme" or name.startswith((".", "~$"))


def sanitize_filename(raw: str | None) -> str | None:
    """把上传的文件名规整成安全的纯文件名；不合法（空、隐藏、~$、README、含控制字符、过长）返回 None。

    只取最后一段（同时按 / 和 \\ 切分），所以 "../../a.pdf" 会变成 "a.pdf"，不可能写出数据目录。
    """
    if not raw:
        return None
    name = raw.replace("\\", "/").split("/")[-1].strip()
    if not name or name in (".", "..") or is_ignored_name(name):
        return None
    if any(ord(char) < 32 or ord(char) == 127 for char in name) or len(name.encode("utf-8")) > 255:
        return None
    return name


def list_documents(data_dir: Path) -> tuple[list[Path], list[Path]]:
    """列出目录（不递归）下所有支持的文档，返回 (文档列表, 被跳过的旧版 .doc 列表)。

    扩展名不区分大小写；README 说明文件、隐藏文件和 Word 临时锁文件（~$开头）不当作文档。
    """
    files, skipped = [], []
    for path in data_dir.iterdir():
        if not path.is_file() or is_ignored_name(path.name):
            continue
        extension = path.suffix.lower()
        if extension in SUPPORTED_EXTS:
            files.append(path)
        elif extension in LEGACY_EXTS:
            skipped.append(path)
    return sorted(files), sorted(skipped)


def report_skipped(skipped: list[Path]) -> None:
    """提示用户被跳过的旧版 .doc 文件。"""
    for path in skipped:
        print(f"警告：跳过旧版 .doc：{path.name}，请另存为 .docx")


# PyMuPDF 不是线程安全的（官方文档的说明）；入库在后台线程里解析，同时可能有别的线程在用，所以串行化
_PDF_LOCK = threading.Lock()
_TEXT_ENCODINGS = ("utf-8-sig", "gb18030")  # utf-8-sig 兼容带 / 不带 BOM 的 UTF-8；都不是再试 GB18030（含 GBK / GB2312）
_PDF_DATE_KEYS = ("creationdate", "moddate")


def _pdf_metadata(doc, path: Path) -> dict:
    """文档级元数据，字段与此前 PyMuPDFLoader 产出的保持一致（已入库的片段带着这些字段）：
    source / file_path / total_pages / producer / creator / creationdate，加上 PDF 自带的字符串 / 整数字段
    （键转小写并去首尾空白；日期转 ISO 格式，同时保留原始 creationDate / modDate）。
    """
    metadata: dict = {
        "producer": "PyMuPDF",
        "creator": "PyMuPDF",
        "creationdate": "",
        "source": str(path),
        "file_path": str(path),
        "total_pages": len(doc),
    }
    raw = {key: value for key, value in doc.metadata.items() if isinstance(value, (str, int))}
    for key, value in raw.items():
        key = key.lstrip("/").lower()
        if key in _PDF_DATE_KEYS:
            try:
                metadata[key] = datetime.strptime(value.replace("'", ""), "D:%Y%m%d%H%M%S%z").isoformat("T")
            except ValueError:
                metadata[key] = value
        elif key in ("page_count", "file_path"):
            continue  # 已有上面规整过的 total_pages / source
        else:
            metadata[key] = value.strip() if isinstance(value, str) else value
    for key in ("modDate", "creationDate"):
        if key in doc.metadata:
            metadata[key] = doc.metadata[key]
    return metadata


def _load_pdf(path: Path) -> list[Document]:
    """PDF：每页一个 Document，正文是 page.get_text() 的默认输出（去首尾空白），page 为 0 起的页码。"""
    import pymupdf

    with _PDF_LOCK, pymupdf.open(path) as doc:
        if doc.is_encrypted and not doc.authenticate(""):
            raise ValueError(f"PDF 已加密，无法解析：{path.name}")
        metadata = _pdf_metadata(doc, path)
        return [
            Document(page_content=page.get_text().strip(), metadata={**metadata, "page": page.number})
            for page in doc
        ]


def _load_docx(path: Path) -> list[Document]:
    """Word（.docx）：整份文档一个 Document，只带 source。"""
    import docx2txt

    return [Document(page_content=docx2txt.process(str(path)), metadata={"source": str(path)})]


def _decode_text(data: bytes, name: str) -> str:
    """字节 -> 文本：UTF-16（带 BOM）、UTF-8（带 / 不带 BOM）、GB18030 依次尝试，都不行则报错。统一换行为 \n。"""
    encodings = ("utf-16",) if data.startswith((b"\xff\xfe", b"\xfe\xff")) else _TEXT_ENCODINGS
    for encoding in encodings:
        try:
            return data.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别文本编码：{name}（已尝试 {'、'.join(encodings)}），请另存为 UTF-8")


def _load_text(path: Path) -> list[Document]:
    """纯文本 / Markdown：整份文件一个 Document，只带 source。"""
    return [Document(page_content=_decode_text(path.read_bytes(), path.name), metadata={"source": str(path)})]


def load_document(path: Path) -> list[Document]:
    """按文件格式解析文档；此层不负责清洗和切分。"""
    path = Path(path)
    extension = path.suffix.lower()
    if extension == ".pdf":
        return _load_pdf(path)
    if extension == ".docx":
        return _load_docx(path)
    if extension in (".md", ".markdown", ".txt"):
        return _load_text(path)
    raise ValueError(f"不支持的格式：{extension}（仅支持 {SUPPORTED_EXTS}）")
