"""RAG 评测指标：Ragas 与结果聚合工具；词面检索指标在 evaluation/lexical.py（此处 re-export，导入路径不变）。"""
import asyncio
import math

from ragas.cache import DiskCacheBackend
from ragas.embeddings import BaseRagasEmbedding
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)

from ragbase.config import Settings, ragas_cache_dir
from ragbase.evaluation.lexical import (  # noqa: F401  re-export：保持 ragbase.evaluation.metrics 的原有导入路径可用
    LEXICAL_HIT_THRESHOLD,
    NORMALIZE_PATTERN,
    POINT_SPLITTER,
    character_ngrams,
    lexical_retrieval_metrics,
    normalize_text,
    point_coverage,
    reference_points,
)
from ragbase.llm.factory import create_ragas_llm


class QwenRagasEmbeddings(BaseRagasEmbedding):
    """把项目已加载的 embedding（本地或云端）适配到 Ragas 0.4 接口。"""

    def __init__(self, embeddings):  # 本地 Qwen3Embeddings 或云端 OpenAIEmbeddings（需有 embed_query / embed_queries）
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


class RagasEvaluator:
    """用配置的评测模型（eval_llm_*，缺省沿用回答模型；OpenAI 兼容协议）运行 Ragas 0.4 指标。"""

    def __init__(self, settings: Settings, embeddings):
        cache = DiskCacheBackend(cache_dir=str(ragas_cache_dir()))
        llm = create_ragas_llm(settings, cache=cache)
        if llm is None:
            raise ValueError("评测大模型未配置：请设置 RAG_LLM_API_KEY / RAG_LLM_MODEL / RAG_LLM_BASE_URL")
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
