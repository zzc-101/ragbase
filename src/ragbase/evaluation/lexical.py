"""词面检索代理指标（纯 Python，不依赖 ragas，导入很快）。

evaluation/metrics.py 会 re-export 这里的名字，原有的导入路径不变；
ragbase diagnose 等只需要词面指标的地方直接导入本模块，避免加载 ragas。
"""
import re

POINT_SPLITTER = re.compile(r"[，,；;。！？!?、：:]")
NORMALIZE_PATTERN = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")
LEXICAL_HIT_THRESHOLD = 0.35


def normalize_text(text: str) -> str:
    return NORMALIZE_PATTERN.sub("", text.lower())


def reference_points(reference: str) -> list[str]:
    """把参考要点拆成可复现的词面匹配单元。"""
    points = [normalize_text(part) for part in POINT_SPLITTER.split(reference)]
    return [point for point in points if len(point) >= 2]


def character_ngrams(text: str, size: int = 2) -> set[str]:
    text = normalize_text(text)
    if len(text) <= size:
        return {text} if text else set()
    return {text[i:i + size] for i in range(len(text) - size + 1)}


def point_coverage(point: str, text: str) -> float:
    """计算一个参考要点的字符二元组在文本中的覆盖率。"""
    point = normalize_text(point)
    text = normalize_text(text)
    if not point or not text:
        return 0.0
    if point in text:
        return 1.0
    expected = character_ngrams(point)
    if not expected:
        return 0.0
    return len(expected & character_ngrams(text)) / len(expected)


def lexical_retrieval_metrics(documents, reference: str) -> dict[str, float]:
    """无需 LLM 的检索代理指标。

    reference 只有答案要点而没有人工相关文档 ID，因此这些是词面代理指标，
    不能冒充有人工标注语料时的标准 Precision/Recall/MRR。
    """
    texts = [doc.page_content for doc in documents]
    points = reference_points(reference)
    if not texts or not points:
        return {
            "point_recall": 0.0,
            "context_precision": 0.0,
            "mrr": 0.0,
        }

    normalized_texts = [normalize_text(text) for text in texts]
    point_ngrams = {point: character_ngrams(point) for point in points}
    text_ngrams = [character_ngrams(text) for text in normalized_texts]

    def is_hit(point: str, text_index: int) -> bool:
        text = normalized_texts[text_index]
        if point in text:
            return True
        expected = point_ngrams[point]
        coverage = len(expected & text_ngrams[text_index]) / len(expected)
        return coverage >= LEXICAL_HIT_THRESHOLD

    point_hits = [
        any(is_hit(point, index) for index in range(len(texts)))
        for point in points
    ]
    relevant_docs = [
        any(is_hit(point, index) for point in points)
        for index in range(len(texts))
    ]
    first_relevant = next(
        (index for index, relevant in enumerate(relevant_docs, 1) if relevant),
        None,
    )
    return {
        "point_recall": sum(point_hits) / len(point_hits),
        "context_precision": sum(relevant_docs) / len(relevant_docs),
        "mrr": 1.0 / first_relevant if first_relevant else 0.0,
    }
