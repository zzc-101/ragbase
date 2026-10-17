"""文档清洗层：规范文本，并过滤重复页眉页脚和空页面。"""
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from langchain_core.documents import Document


CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HORIZONTAL_WHITESPACE = re.compile(r"[ \t\u00a0\u3000]+")
MULTIPLE_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass
class CleaningReport:
    input_documents: int = 0
    output_documents: int = 0
    empty_documents_removed: int = 0
    repeated_boilerplate_lines_removed: int = 0
    repeated_boilerplate_patterns: int = 0


def normalize_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line)
    line = CONTROL_CHARACTERS.sub("", line)
    return HORIZONTAL_WHITESPACE.sub(" ", line).strip()


def normalize_text(text: str) -> str:
    """规范 Unicode、控制字符、横向空白和连续空行。"""
    lines = [normalize_line(line) for line in text.splitlines()]
    return MULTIPLE_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def line_key(line: str) -> str:
    return re.sub(r"\s+", "", normalize_line(line)).lower()


def find_repeated_boilerplate_lines(documents: list[Document]) -> set[str]:
    """识别跨大量页面重复的短模板行，如页眉、页脚和书名。"""
    if len(documents) < 3:
        return set()

    occurrences = Counter()
    for document in documents:
        lines = [line for line in document.page_content.splitlines() if line.strip()]
        occurrences.update({line_key(line) for line in lines if line_key(line)})

    threshold = max(3, math.ceil(len(documents) * 0.3))
    return {
        key
        for key, count in occurrences.items()
        if count >= threshold and 4 <= len(key) <= 80
    }


def clean_documents(
    documents: list[Document],
) -> tuple[list[Document], CleaningReport]:
    """按文档集合清洗内容，同时返回可观测的清洗报告。"""
    report = CleaningReport(input_documents=len(documents))
    repeated_lines = find_repeated_boilerplate_lines(documents)
    report.repeated_boilerplate_patterns = len(repeated_lines)
    cleaned_documents = []

    for document in documents:
        raw_lines = document.page_content.splitlines()
        kept_lines = []
        for line in raw_lines:
            if line_key(line) in repeated_lines:
                report.repeated_boilerplate_lines_removed += 1
                continue
            kept_lines.append(line)

        content = normalize_text("\n".join(kept_lines))
        if not content:
            report.empty_documents_removed += 1
            continue

        metadata = dict(document.metadata)
        metadata["cleaned"] = True
        cleaned_documents.append(Document(page_content=content, metadata=metadata))

    report.output_documents = len(cleaned_documents)
    return cleaned_documents, report
