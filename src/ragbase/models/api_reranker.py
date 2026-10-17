"""云端 rerank：通用的 POST {base}/rerank，请求 {model, query, documents, top_n}，响应 {results: [{index, relevance_score}]}
（Jina、SiliconFlow、Cohere v2 兼容网关、Xinference、vLLM 等的事实标准形状）。

与本地 Qwen3Reranker 同一套接口（rerank / rerank_with_scores）。分数原样透传（多数服务是 0~1），
所以 RAG_MIN_SCORE / RAG_CONTEXT_MIN_SCORE 必须按这个 reranker 重新标定（scripts/calibrate_rerank_threshold.py）。
"""
import math

from ragbase.models.api_client import ApiClient, ProviderError

CACHE_SIZE = 128  # 进程内缓存的 (问题, 候选集, top_k) 条数上限


class ApiReranker:
    def __init__(self, client: ApiClient, model: str):
        self._client = client
        self.model = model
        self._cache: dict[tuple[str, tuple[str, ...], int], tuple[tuple[int, float], ...]] = {}

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[int]:
        """按相关性重排，返回 top_k 个文档的原始索引（相关性从高到低）。"""
        return [index for index, _ in self.rerank_with_scores(query, documents, top_k)]

    def rerank_with_scores(self, query: str, documents: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        """返回 [(原始索引, 服务端给的相关度分数)]，分数从高到低，至多 top_k 条。"""
        if not documents:
            return []
        top_k = min(top_k, len(documents))
        key = (query, tuple(documents), top_k)
        ranked = self._cache.get(key)
        if ranked is None:
            ranked = tuple(self._request(query, documents, top_k))
            if len(self._cache) >= CACHE_SIZE:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = ranked
        return list(ranked)

    def _request(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        data = self._client.post("rerank", {"model": self.model, "query": query, "documents": documents, "top_n": top_k})
        results = data.get("results")
        if not isinstance(results, list):
            raise ProviderError("rerank 接口返回缺少 results 数组；请确认地址与服务支持 POST {base}/rerank")
        scored: dict[int, float] = {}
        for item in results:
            try:
                index = int(item["index"])
                score = float(item["relevance_score"] if "relevance_score" in item else item["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProviderError("rerank 接口返回的结果项需要 index 与 relevance_score 字段") from exc
            if not 0 <= index < len(documents) or not math.isfinite(score):
                raise ProviderError(f"rerank 接口返回了无效的结果项（index={index}，候选 {len(documents)} 个）")
            scored.setdefault(index, score)
        # 按分数降序（分数相同保持服务端顺序）；有的服务忽略 top_n 会返回全部，这里统一截断
        ordered = sorted(scored.items(), key=lambda pair: pair[1], reverse=True)
        return ordered[:top_k]

    def probe(self) -> None:
        """发一条最小请求（doctor --check-models 用）。"""
        self._request("ping", ["pong", "ping"], 1)


class NoReranker:
    """provider=none：不精排，保持召回（RRF / 向量）顺序取前 top_k；没有相关度分数（None），所以不会按得分拒答。"""

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[int]:
        return list(range(min(top_k, len(documents))))

    def rerank_with_scores(self, query: str, documents: list[str], top_k: int = 5) -> list[tuple[int, None]]:
        return [(index, None) for index in self.rerank(query, documents, top_k)]
