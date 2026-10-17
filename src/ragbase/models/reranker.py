"""Reranker：封装 Qwen3-Reranker（默认 0.6B，可用 RAG_RERANKER_MODEL 更换；Cross-Encoder）。

用法：对初步召回的候选片段做精排，把最相关的排前面。
Cross-Encoder 把 query 和每个 document 拼一起送进模型，输出相关性分数。
"""
import math

from sentence_transformers import CrossEncoder

from ragbase.config import HALF_PRECISION_DEVICES, get_settings, resolve_device

CACHE_SIZE = 128  # 进程内缓存的 (问题, 候选集) 条数上限


class Qwen3Reranker:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 8,
    ):
        # model_name / device 为空时按配置解析（RAG_RERANKER_MODEL，默认 Qwen/Qwen3-Reranker-0.6B；RAG_DEVICE，默认 auto：mps > cuda > cpu）
        settings = get_settings()
        model_name = model_name or settings.reranker_model
        device = resolve_device(device or settings.device)
        self.device = device
        model_kwargs = {"torch_dtype": "float16"} if device in HALF_PRECISION_DEVICES else None
        load_kwargs = {
            "device": device,
            "model_kwargs": model_kwargs,
        }
        try:
            self._model = CrossEncoder(
                model_name,
                local_files_only=True,
                **load_kwargs,
            )
        except OSError:
            self._model = CrossEncoder(model_name, **load_kwargs)
        self._batch_size = batch_size
        # 缓存相同问题和候选集的分数；不用 lru_cache 装饰实例方法（会让缓存持有 self）
        self._cache: dict[tuple[str, tuple[str, ...]], tuple[float, ...]] = {}
        if device == "mps":
            # 提前触发 MPS kernel 编译，避免首个真实问题额外等待约 2 秒。
            self._model.predict(
                [("__warmup__", "预热文本")] * batch_size,
                batch_size=batch_size,
                show_progress_bar=False,
            )

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[int]:
        """对 documents 按与 query 的相关性重排，返回 top_k 个文档的原始索引。

        Args:
            query: 用户问题
            documents: 候选文档文本列表
            top_k: 取前几个

        Returns:
            排序后的文档索引列表（相关性从高到低），长度 min(top_k, len(documents))
        """
        return [index for index, _ in self.rerank_with_scores(query, documents, top_k)]

    def rerank_with_scores(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """同 rerank，但同时返回相关概率：[(原始索引, 分数)]，分数从高到低。

        分数是 P(相关) ∈ [0, 1]。模型的 predict 实际返回原始 logit 差
        （"yes" 与 "no" 两个 token 的 logit 之差，activation 为 Identity，
        实测相关对约 +9~+10、无关对约 -4~-11），这里用 sigmoid 转成概率。
        """
        if not documents:
            return []
        key = (query, tuple(documents))
        scores = self._cache.get(key)
        if scores is None:
            pairs = [(query, document) for document in documents]
            logits = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
            scores = tuple(_sigmoid(float(logit)) for logit in logits)
            if len(self._cache) >= CACHE_SIZE:
                self._cache.pop(next(iter(self._cache)))  # 淘汰最早放入的一条
            self._cache[key] = scores
        # 按分数从高到低排序，取 top_k 的原始索引
        ranked = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in ranked[:top_k]]


def _sigmoid(x: float) -> float:
    """数值稳定的 sigmoid：把 logit 转成 0~1 的概率。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)
