"""ragbase.ingest.toc 的单元测试：用 fitz 在临时目录生成带书签的小 PDF（不加载模型、不访问网络）。

运行：uv run python tests/test_toc.py
"""
import tempfile
from pathlib import Path

import pymupdf

from ragbase.ingest.toc import build_page_headings, clean_title, clean_toc, pdf_page_headings
from ragbase.ingest.common import bm25_input, embedding_input, prepare_file_chunks


def _make_pdf(path: Path, pages: int, toc: list | None, text_prefix: str = "正文"):
    document = pymupdf.open()
    for number in range(1, pages + 1):
        page = document.new_page()
        page.insert_text((72, 100), f"{text_prefix}{number}", fontname="china-s")
    if toc is not None:
        document.set_toc(toc)
    document.save(str(path))
    document.close()


def test_clean_title():
    assert clean_title("  第11章 \n 项目　成本   管理 ") == "第11章 项目 成本 管理"
    assert clean_title(None) == "" and clean_title("   ") == ""


def test_clean_toc_drops_numeric_and_invalid():
    toc = [
        [1, "1", 1],            # 纯数字：丢弃
        [1, " 3.1 ", 2],        # 数字 + 标点：丢弃
        [1, "（二）", 2],        # 汉字：保留
        [1, "", 3],             # 空：丢弃
        [1, "附录 A", -1],      # 页码无效：丢弃
        [2, "ISO 9001", 5],     # 字母 + 数字：保留
    ]
    assert clean_toc(toc) == [(1, "（二）", 2), (2, "ISO 9001", 5)]


def test_build_page_headings_hierarchy():
    toc = [
        [1, "第1章 绪论", 1],
        [2, "1.1 背景", 1],
        [2, "1.2 意义", 3],
        [3, "1.2.1 细节", 4],
        [1, "第2章 方法", 5],
        [2, "2.1 总述", 6],
    ]
    headings = build_page_headings(toc, 7)
    assert headings == [
        "第1章 绪论 > 1.1 背景",                 # 第 1 页
        "第1章 绪论 > 1.1 背景",                 # 第 2 页
        "第1章 绪论 > 1.2 意义",                 # 第 3 页：同层替换
        "第1章 绪论 > 1.2 意义 > 1.2.1 细节",    # 第 4 页：更深一层
        "第2章 方法",                            # 第 5 页：新章清掉旧的 2、3 级
        "第2章 方法 > 2.1 总述",
        "第2章 方法 > 2.1 总述",
    ], headings


def test_build_page_headings_before_first_entry_and_numeric_parent():
    toc = [[1, "7", 1], [2, "有用的节", 3], [1, "第二部分", 5]]
    headings = build_page_headings(toc, 6)
    # 第 1~2 页还没有任何有效书签；纯数字的一级标题被丢弃后，二级标题单独成路径
    assert headings == ["", "", "有用的节", "有用的节", "第二部分", "第二部分"], headings
    assert build_page_headings([], 3) == ["", "", ""]


def test_build_page_headings_unsorted_toc_and_out_of_range():
    toc = [[1, "后面的章", 4], [1, "前面的章", 1], [1, "越界的章", 99]]
    assert build_page_headings(toc, 5) == ["前面的章"] * 3 + ["后面的章"] * 2


def test_pdf_page_headings_from_real_pdf():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "book.pdf"
        _make_pdf(path, 4, [[1, "第1章 概述", 1], [2, "1.1 范围", 2], [1, "2", 3], [1, "第2章 方法", 4]])
        assert pdf_page_headings(path) == [
            "第1章 概述", "第1章 概述 > 1.1 范围", "第1章 概述 > 1.1 范围", "第2章 方法",
        ]


def test_pdf_without_toc_or_only_useless_titles_has_no_headings():
    with tempfile.TemporaryDirectory() as tmp:
        no_toc = Path(tmp) / "plain.pdf"
        _make_pdf(no_toc, 2, None)
        assert pdf_page_headings(no_toc) == []
        numeric = Path(tmp) / "numeric.pdf"
        _make_pdf(numeric, 3, [[1, "1", 1], [1, "2", 2], [1, "3.1", 3]])
        assert pdf_page_headings(numeric) == []


def test_broken_pdf_returns_empty_instead_of_raising():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "broken.pdf"
        path.write_bytes(b"not a pdf")
        assert pdf_page_headings(path) == []


def test_prepare_file_chunks_attaches_heading_path_for_pdf_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pdf = root / "book.pdf"
        _make_pdf(pdf, 3, [[1, "第1章 概述", 1], [2, "1.1 范围", 2]], text_prefix="项目管理内容")
        texts, metadatas, ids, report = prepare_file_chunks(pdf)
        assert len(texts) == 3
        by_page = {metadata["page"]: metadata.get("heading_path") for metadata in metadatas}
        assert by_page == {0: "第1章 概述", 1: "第1章 概述 > 1.1 范围", 2: "第1章 概述 > 1.1 范围"}, by_page
        assert report["heading_chunks"] == 3
        assert all(isinstance(value, (str, int, float, bool)) for metadata in metadatas for value in metadata.values())
        assert embedding_input(texts[1], metadatas[1]).startswith("第1章 概述 > 1.1 范围\n")
        assert bm25_input(texts[1], metadatas[1]).startswith("第1章 概述 > 1.1 范围 ")

        plain = root / "plain.pdf"
        _make_pdf(plain, 2, None)
        _, plain_metadatas, _, plain_report = prepare_file_chunks(plain)
        assert all("heading_path" not in metadata for metadata in plain_metadatas)  # 没有书签：字段干脆不写
        assert plain_report["heading_chunks"] == 0

        note = root / "note.md"
        note.write_text("# 标题\n\n正文内容", encoding="utf-8")
        _, md_metadatas, _, _ = prepare_file_chunks(note)
        assert all("heading_path" not in metadata for metadata in md_metadatas)  # 非 PDF 暂无章节
        assert embedding_input("正文", {}) == "正文" and bm25_input("正文", {"heading_path": ""}) == "正文"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
