"""云端 embedding：OpenAI 兼容的 POST {base}/embeddings（OpenAI、SiliconFlow、DashScope 兼容模式、火山方舟、Ollama、vLLM、Xinference 等）。

与本地 Qwen3Embeddings 同一套接口（embed_documents / embed_query / embed_queries），可互换：
- 返回的向量统一做 L2 归一化，保证 Chroma 的 L2 距离排序与本地模型（normalize_embeddings=True）一致；
- 文本按 batch_size 分批请求；可选 dimensions 参数；可选只加在查询前面的指令前缀；
- 查询向量在进程内缓存（与本地实现一样），重复问题不重复调用接口。
"""
import math
from collections import OrderedDict

from langchain_core.embeddings import Embeddings

from ragbase.models.api_client import ApiClient, ProviderError

QUERY_CACHE_SIZE = 256


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm > 0 else vector


class OpenAIEmbeddings(Embeddings):
    def __init__(
        self,
        client: ApiClient,
        model: str,
        dimensions: int | None = None,
        batch_size: int = 16,
        query_instruction: str = "",
    ):
        self._client = client
        self.model = model
        self.dimensions = dimensions
        self._batch_size = max(1, batch_size)
        self.query_instruction = query_instruction
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload: dict = {"model": self.model, "input": texts}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        data = self._client.post("embeddings", payload)
        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(texts):
            got = len(items) if isinstance(items, list) else "无"
            raise ProviderError(f"embedding 接口返回的向量数量不对（请求 {len(texts)} 条，返回 {got}）；响应缺少 data 数组？")
        try:
            ordered = sorted(items, key=lambda item: item.get("index", 0))  # 规范要求带 index，缺省按原顺序
            vectors = [[float(x) for x in item["embedding"]] for item in ordered]
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ProviderError("embedding 接口返回的数据格式不符合 OpenAI 规范（每项需要 embedding 数组；不要使用 base64 编码的服务）") from exc
        lengths = {len(vector) for vector in vectors}
        if len(lengths) != 1 or 0 in lengths:
            raise ProviderError(f"embedding 接口返回的向量长度不一致或为空：{sorted(lengths)}")
        if self.dimensions and lengths != {self.dimensions}:
            raise ProviderError(
                f"embedding 接口返回的向量长度是 {lengths.pop()}，与 RAG_EMBEDDING_DIMENSIONS={self.dimensions} 不符"
                "（该服务可能不支持 dimensions 参数；请清空此设置，或换用支持的模型）"
            )
        return [_normalize(vector) for vector in vectors]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(texts[start:start + self._batch_size]))
        return vectors

    def embed_documents(self, texts):
        """文档编码：不加查询指令，按批请求。"""
        return self._embed(list(texts))

    def embed_queries(self, texts):
        """批量编码查询（加指令前缀），供检索和 Ragas 语义相关性评测复用。"""
        return self._embed([self.query_instruction + text for text in texts])

    def embed_query(self, text):
        """查询编码：加指令前缀；进程内缓存重复问题。"""
        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return list(cached)
        vector = self.embed_queries([text])[0]
        self._cache[text] = tuple(vector)
        if len(self._cache) > QUERY_CACHE_SIZE:
            self._cache.popitem(last=False)
        return vector

    def probe(self) -> int:
        """发一条最小请求，返回向量长度（doctor --check-models 用）。"""
        return len(self._embed(["ping"])[0])
