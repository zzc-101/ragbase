"""document_parser 的单元测试：用临时生成的 PDF / DOCX / 文本文件，不依赖 data/ 里的真实文档。

DOCX 没有用 python-docx（项目不依赖它），而是手工打包最小的 .docx（zip + document.xml），docx2txt 能照常解析。

运行：uv run python tests/test_parser.py
"""
import re
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf

from document_parser import load_document

ROOT = Path(__file__).resolve().parent.parent
TMP = tempfile.TemporaryDirectory(prefix="ragdemo-parser-test-")
DIR = Path(TMP.name)


def _make_pdf(path: Path, pages: list[str], metadata: dict | None = None) -> Path:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    if metadata:
        doc.set_metadata(metadata)
    doc.save(path)
    doc.close()
    return path


def _make_docx(path: Path, paragraphs: list[str]) -> Path:
    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("word/document.xml", document)
    return path


def test_pdf_one_document_per_page_with_loader_compatible_metadata():
    path = _make_pdf(
        DIR / "book.pdf", ["Chapter one", "", "Third page  "],
        {"title": "  测试书  ", "author": "作者", "creationDate": "D:20240102030405+08'00'"},
    )
    docs = load_document(path)
    assert [doc.metadata["page"] for doc in docs] == [0, 1, 2]  # 0 起，每页一个 Document（空页也保留）
    assert docs[0].page_content == "Chapter one" and docs[1].page_content == "" and docs[2].page_content == "Third page"
    meta = docs[0].metadata
    assert meta["source"] == meta["file_path"] == str(path) and meta["total_pages"] == 3
    assert meta["title"] == "测试书" and meta["author"] == "作者"  # 键小写、值去首尾空白
    assert meta["creationdate"] == "2024-01-02T03:04:05+08:00" and meta["creationDate"] == "D:20240102030405+08'00'"
    assert "producer" in meta and "creator" in meta
    # Chroma 元数据只能是标量
    assert all(isinstance(value, (str, int)) for doc in docs for value in doc.metadata.values())
    # 每页的文档级字段一致，只有 page 不同
    assert all({k: v for k, v in d.metadata.items() if k != "page"} == {k: v for k, v in meta.items() if k != "page"} for d in docs)


def test_pdf_page_text_matches_pymupdf_default_get_text():
    path = _make_pdf(DIR / "plain.pdf", ["第一页 hello", "second page"])
    with pymupdf.open(path) as doc:
        expected = [page.get_text().strip() for page in doc]
    assert [d.page_content for d in load_document(path)] == expected
    assert "hello" in expected[0] and "second" in expected[1]


def test_pdf_without_info_still_has_standard_metadata():
    docs = load_document(_make_pdf(DIR / "bare.pdf", ["x"]))
    assert docs[0].metadata["total_pages"] == 1 and docs[0].metadata["page"] == 0
    assert "source" in docs[0].metadata and "creationdate" in docs[0].metadata


def test_pdf_encrypted_without_password_is_rejected():
    plain = _make_pdf(DIR / "to-encrypt.pdf", ["secret"])
    locked = DIR / "locked.pdf"
    with pymupdf.open(plain) as doc:
        doc.save(locked, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="owner")
    try:
        load_document(locked)
    except ValueError as exc:
        assert "加密" in str(exc) and "locked.pdf" in str(exc)
    else:
        raise AssertionError("加密 PDF 应报错")


def test_docx_single_document_with_source_only():
    path = _make_docx(DIR / "notes.docx", ["第一段 项目管理", "Second paragraph"])
    docs = load_document(path)
    assert len(docs) == 1 and docs[0].metadata == {"source": str(path)}
    assert "第一段 项目管理" in docs[0].page_content and "Second paragraph" in docs[0].page_content


def test_text_encodings_utf8_bom_gb18030_utf16():
    content = "软件工程\n第二行：需求分析"
    cases = {
        "utf8.txt": content.encode("utf-8"),
        "bom.md": b"\xef\xbb\xbf" + content.encode("utf-8"),
        "gbk.txt": content.encode("gb18030"),
        "crlf.markdown": content.replace("\n", "\r\n").encode("utf-8"),
        "utf16.txt": content.encode("utf-16"),
    }
    for name, data in cases.items():
        path = DIR / name
        path.write_bytes(data)
        docs = load_document(path)
        assert len(docs) == 1 and docs[0].page_content == content, (name, docs[0].page_content)
        assert docs[0].metadata == {"source": str(path)}


def test_text_undecodable_raises_clear_error():
    path = DIR / "binary.txt"
    path.write_bytes(b"\x81\x20\x81\x20\xff\xff\xfe")  # 既不是 UTF-8 也不是合法 GB18030
    try:
        load_document(path)
    except ValueError as exc:
        assert "编码" in str(exc) and "binary.txt" in str(exc)
    else:
        raise AssertionError("无法解码应报错")


def test_unsupported_extension_and_str_path():
    try:
        load_document(DIR / "x.doc")
    except ValueError as exc:
        assert "不支持" in str(exc)
    else:
        raise AssertionError("应拒绝 .doc")
    text = DIR / "str.txt"
    text.write_text("abc", encoding="utf-8")
    assert load_document(str(text))[0].page_content == "abc"  # 也接受 str 路径


def test_project_does_not_import_langchain_community_or_construct_chatopenai_elsewhere():
    pattern_community = re.compile(r"^\s*(?:from|import)\s+langchain_community", re.MULTILINE)
    offenders, direct = [], []
    for source in ROOT.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        if pattern_community.search(text):
            offenders.append(source.name)
        if "ChatOpenAI(" in text and source.name != "llm_factory.py":
            direct.append(source.name)
    assert not offenders, f"仍在导入 langchain_community：{offenders}"
    assert not direct, f"llm_factory 之外直接构造了 ChatOpenAI：{direct}"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
