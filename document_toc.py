"""PDF 书签目录 -> 每页的章节路径（heading_path）。

PDF 的书签（fitz 的 get_toc）形如 [层级, 标题, 起始页（1 起）]。对 0 起的第 p 页，
路径 = 起始页 <= p+1 的各层级“最近一条”书签标题，用 " > " 连接，例如：
    第11章 项目成本管理 > 11.3 挣值分析

近似说明：书签只记录起始页，不记录页内位置，所以一节从页中间开始时，整页都归给这一节
（该页上位于它之前的内容会被标上新章节）。这是粗粒度的章节定位，不是精确的段落归属。
"""
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

HEADING_SEPARATOR = " > "
_HAS_LETTER = re.compile(r"[^\W\d_]")  # Unicode 字母（含汉字）；纯数字 / 标点标题不含它


def clean_title(title: str) -> str:
    """去首尾空白并把连续空白（含全角空格、换行）压成一个空格。"""
    return re.sub(r"\s+", " ", title or "").strip()


def clean_toc(toc: list) -> list[tuple[int, str, int]]:
    """清洗书签：规范标题，丢弃纯数字 / 标点标题（如 "1"、"3.1"）和无效页码，按起始页稳定排序。"""
    entries = []
    for level, title, page in toc:
        title = clean_title(title)
        if not title or not _HAS_LETTER.search(title):
            continue
        if not isinstance(page, int) or page < 1 or not isinstance(level, int) or level < 1:
            continue
        entries.append((level, title, page))
    entries.sort(key=lambda entry: entry[2])  # 稳定排序：同一页的先后顺序保持书签原顺序
    return entries


def build_page_headings(toc: list, page_count: int) -> list[str]:
    """返回长度为 page_count 的列表，第 p 项是 0 起第 p 页的章节路径（没有则为空串）。

    新书签出现时，同层及更深层的旧标题被它替换 / 清除，所以路径始终是一条自洽的层级链。
    """
    entries = clean_toc(toc)
    stack: list[tuple[int, str]] = []
    headings = []
    index = 0
    for page in range(1, page_count + 1):
        while index < len(entries) and entries[index][2] <= page:
            level, title, _ = entries[index]
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            index += 1
        headings.append(HEADING_SEPARATOR.join(title for _, title in stack))
    return headings


def pdf_page_headings(path: Path) -> list[str]:
    """读取 PDF 书签并生成每页章节路径；没有书签、书签全是无用标题或文件打不开时返回空列表。"""
    import pymupdf

    try:
        with pymupdf.open(str(path)) as document:
            headings = build_page_headings(document.get_toc(simple=True), document.page_count)
    except Exception as exc:  # 书签只是增强信息，读不出来不应让入库失败
        logger.warning("读取 PDF 书签失败，跳过章节标题：%s（%s）", Path(path).name, type(exc).__name__)
        return []
    return headings if any(headings) else []
