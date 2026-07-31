"""Reranker：封装 Qwen3-Reranker-0.6B（Cross-Encoder）。

用法：对初步召回的候选片段做精排，把最相关的排前面。
Cross-Encoder 把 query 和每个 document 拼一起送进模型，输出相关性分数。
"""
from sentence_transformers import CrossEncoder


class Qwen3Reranker:
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B", device: str = "mps"):
        self._model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[int]:
        """对 documents 按与 query 的相关性重排，返回 top_k 个文档的原始索引。

        Args:
            query: 用户问题
            documents: 候选文档文本列表
            top_k: 取前几个

        Returns:
            排序后的文档索引列表（相关性从高到低），长度 min(top_k, len(documents))
        """
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)
        # 按分数从高到低排序，取 top_k 的原始索引
        ranked = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
        return ranked[:top_k]
