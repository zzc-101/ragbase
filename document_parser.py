"""文档解析层：负责把不同文件格式转换为 LangChain Document。"""
from pathlib import Path


SUPPORTED_EXTS = {".pdf", ".docx", ".md", ".markdown", ".txt"}
LEGACY_EXTS = {".doc"}  # 旧版二进制 Word，Docx2txtLoader 读不了，只提示不入库


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


def load_document(path: Path):
    """按文件格式解析文档；此层不负责清洗和切分。"""
    extension = path.suffix.lower()
    if extension == ".pdf":
        from langchain_community.document_loaders import PyMuPDFLoader

        loader = PyMuPDFLoader(str(path))
    elif extension == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        loader = Docx2txtLoader(str(path))
    elif extension in (".md", ".markdown", ".txt"):
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(
            str(path),
            encoding="utf-8",
            autodetect_encoding=True,
        )
    else:
        raise ValueError(f"不支持的格式：{extension}（仅支持 {SUPPORTED_EXTS}）")
    return loader.load()
