"""按配置创建向量库：RAG_VECTOR_STORE=chroma（默认）| qdrant。"""
from ragbase.config import VECTOR_STORES, Settings, get_settings
from ragbase.vectorstores.base import VectorStore


def create_vector_store(settings: Settings | None = None, backend: str | None = None) -> VectorStore:
    """创建（延迟打开的）向量库对象。backend 缺省取 settings.vector_store；未知取值抛 ValueError。

    创建对象不会建目录 / 连接服务；本地 Qdrant 在进程内按目录共享同一个 client（见 qdrant.py）。
    """
    settings = settings or get_settings()
    backend = (backend or settings.vector_store).strip().lower()
    if backend == "chroma":
        from ragbase.vectorstores.chroma import ChromaStore

        return ChromaStore(settings.db_dir, settings.collection)
    if backend == "qdrant":
        from ragbase.vectorstores.qdrant import QdrantStore

        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        return QdrantStore(
            settings.collection,
            path=settings.qdrant_path,
            url=settings.qdrant_url,
            api_key=api_key,
        )
    raise ValueError(f"未知的向量库 {backend!r}（RAG_VECTOR_STORE）；可选：{'、'.join(VECTOR_STORES)}")


def store_count(settings: Settings | None = None) -> int | None:
    """索引里的片段数；索引不存在（目录 / 集合不存在或连接失败）返回 None。不加载任何模型，也不创建任何东西。"""
    try:
        store = create_vector_store(settings)
        return store.count() if store.exists() else None
    except Exception:
        return None
