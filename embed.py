"""自定义 embedding：封装 Qwen3-Embedding-0.6B。

Qwen3 是 instruction-aware 模型，查询和文档需要不同的编码方式：
- 查询（query）：加 query prompt（prompt_name="query"）
- 文档（document）：直接编码，不加 prompt

langchain 的 HuggingFaceEmbeddings 对 query/document 用同一套 encode_kwargs，
无法分别处理，所以这里继承 Embeddings 基类自己封装。
"""
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class Qwen3Embeddings(Embeddings):
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "mps"):
        # torch_dtype=fp16 半精度，省显存、加速；M 芯片 MPS 支持
        self._model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs={"torch_dtype": "float16"},
        )
        self._batch_size = 64  # 批量编码，M 芯片这个值较稳

    def embed_documents(self, texts):
        """文档编码：不加 query prompt，批量并行。"""
        return self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,  # 显示进度条
        ).tolist()

    def embed_query(self, text):
        """查询编码：加 query prompt。"""
        return self._model.encode(
            [text], prompt_name="query", normalize_embeddings=True
        ).tolist()[0]
