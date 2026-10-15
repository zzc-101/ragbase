"""RAG 评测指标：Ragas、词面检索指标及结果聚合工具。"""
import asyncio
import math
import re
from pathlib import Path

from openai import AsyncOpenAI
from ragas.cache import DiskCacheBackend
from ragas.embeddings import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

from embed import Qwen3Embeddings


BASE_DIR = Path(__file__).parent
RAGAS_CACHE_DIR = BASE_DIR / ".ragas_cache"
POINT_SPLITTER = re.compile(r"[，,；;。！？!?、：:]")
NORMALIZE_PATTERN = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")
LEXICAL_HIT_THRESHOLD = 0.35


class QwenRagasEmbeddings(BaseRagasEmbedding):
    """把项目已加载的 Qwen embedding 适配到 Ragas 0.4 接口。"""

    def __init__(self, embeddings: Qwen3Embeddings):
        super().__init__()
        self.embeddings = embeddings

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self.embeddings.embed_query(text)

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text, **kwargs)

    def embed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
        return self.embeddings.embed_queries(texts)

    async def aembed_texts(
        self,
        texts: list[str],
        **kwargs,
    ) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_texts, texts, **kwargs)


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


class RagasEvaluator:
    """使用方舟 OpenAI-compatible 接口运行 Ragas 0.4 指标。"""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        embeddings: Qwen3Embeddings,
    ):
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        cache = DiskCacheBackend(cache_dir=str(RAGAS_CACHE_DIR))
        llm = llm_factory(
            model=model,
            provider="openai",
            client=client,
            cache=cache,
            temperature=0,
        )
        ragas_embeddings = QwenRagasEmbeddings(embeddings)
        self.metrics = {
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(
                llm=llm,
                embeddings=ragas_embeddings,
                strictness=1,
            ),
            "factual_correctness": FactualCorrectness(llm=llm),
            "context_precision": ContextPrecisionWithReference(llm=llm),
            "context_recall": ContextRecall(llm=llm),
        }

    async def score(
        self,
        question: str,
        response: str,
        retrieved_contexts: list[str],
        reference: str,
    ) -> tuple[dict[str, float | None], dict[str, str]]:
        calls = {
            "faithfulness": self.metrics["faithfulness"].ascore(
                user_input=question,
                response=response,
                retrieved_contexts=retrieved_contexts,
            ),
            "answer_relevancy": self.metrics["answer_relevancy"].ascore(
                user_input=question,
                response=response,
            ),
            "factual_correctness": self.metrics["factual_correctness"].ascore(
                response=response,
                reference=reference,
            ),
            "context_precision": self.metrics["context_precision"].ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            ),
            "context_recall": self.metrics["context_recall"].ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            ),
        }
        names = list(calls)
        outcomes = await asyncio.gather(
            *calls.values(),
            return_exceptions=True,
        )

        scores: dict[str, float | None] = {}
        errors: dict[str, str] = {}
        for name, outcome in zip(names, outcomes):
            if isinstance(outcome, BaseException):
                scores[name] = None
                errors[name] = f"{type(outcome).__name__}: {outcome}"
                continue
            value = float(outcome.value)
            scores[name] = None if math.isnan(value) else round(value, 4)
        return scores, errors


def mean_scores(results: list[dict], path: tuple[str, ...]) -> dict[str, float]:
    """按嵌套路径聚合数值字段，自动忽略 None。"""
    rows = []
    for result in results:
        value = result
        for key in path:
            if not isinstance(value, dict):
                value = {}
                break
            value = value.get(key, {})
        if isinstance(value, dict):
            rows.append(value)

    names = sorted({name for row in rows for name in row})
    means = {}
    for name in names:
        values = [
            row[name]
            for row in rows
            if isinstance(row.get(name), (int, float))
        ]
        if values:
            means[name] = round(sum(values) / len(values), 4)
    return means
