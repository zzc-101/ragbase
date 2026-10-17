"""Chroma 后端：行为与重构前直接使用 chromadb 完全一致（同样的集合名、默认 L2 空间、where 语义）。

这是整个项目里唯一 import chromadb 的地方。Chroma 的 PersistentClient 会自动创建目录，
所以“只读探测”（count / exists / query）在目录或集合不存在时不会创建任何东西。
"""
import logging
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

from ragbase.vectorstores.base import ITER_BATCH_SIZE, Hit, Record

logger = logging.getLogger(__name__)

_DELETE_BATCH = 1000  # 每次 delete 的 ID 数（低于 SQLite 变量上限）


class ChromaStore:
    name = "chroma"

    def __init__(self, path: str | Path, collection: str = "rag"):
        self.path = Path(path)
        self.collection_name = collection
        self.location = str(self.path)
        self._client = None
        self._collection = None
        self._lock = threading.RLock()

    # ---- 内部：延迟打开 client / collection ------------------------------------------------

    def _get_client(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self.path))
        return self._client

    def _open(self, create: bool):
        """返回 collection 对象；不存在且不允许创建时返回 None（只读探测不会建目录）。"""
        with self._lock:
            if self._collection is not None:
                return self._collection
            if not create and not self.path.exists():  # PersistentClient 会自动建目录，所以先判断
                return None
            client = self._get_client()
            if create:
                self._collection = client.get_or_create_collection(self.collection_name)
            else:
                try:
                    self._collection = client.get_collection(self.collection_name)
                except Exception:  # 集合不存在（异常类型随 chromadb 版本而变）
                    logger.debug("Chroma 集合 %s 不存在", self.collection_name)
                    return None
            return self._collection

    # ---- VectorStore ------------------------------------------------------------------------

    def exists(self) -> bool:
        return self._open(create=False) is not None

    def count(self) -> int:
        collection = self._open(create=False)
        return collection.count() if collection is not None else 0

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        if not ids:
            return
        self._open(create=True).upsert(
            ids=list(ids),
            embeddings=[list(map(float, vector)) for vector in embeddings],
            documents=list(documents),
            metadatas=list(metadatas),
        )

    def delete(self, ids: Sequence[str]) -> None:
        collection = self._open(create=False)
        if collection is None:
            return
        ids = list(ids)
        for start in range(0, len(ids), _DELETE_BATCH):
            collection.delete(ids=ids[start:start + _DELETE_BATCH])

    def _ids(self, where: dict | None) -> list[str]:
        collection = self._open(create=False)
        if collection is None:
            return []
        options = {} if where is None else {"where": where}
        return list(collection.get(include=[], **options)["ids"])

    def ids_where(self, fid: str | None = None, sources: Sequence[str] | None = None) -> list[str]:
        ids: dict[str, None] = {}
        if fid:
            ids.update(dict.fromkeys(self._ids({"fid": fid})))
        if sources:
            ids.update(dict.fromkeys(self._ids({"source": {"$in": list(sources)}})))
        return list(ids)

    def all_ids(self) -> list[str]:
        return self._ids(None)

    def query(self, embedding: Sequence[float], k: int, sources: Sequence[str] | None = None) -> list[Hit]:
        collection = self._open(create=False)
        if collection is None or k <= 0:
            return []
        options = {}
        if sources is not None:
            if not sources:
                return []
            options["where"] = {"source": {"$in": list(sources)}}
        result = collection.query(
            query_embeddings=[list(map(float, embedding))],
            n_results=k,
            include=["documents", "metadatas", "distances"],
            **options,
        )
        return [
            Hit(id=doc_id, text=text or "", metadata=metadata or {}, score=float(distance))
            for doc_id, text, metadata, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def iter_all(self, batch_size: int = ITER_BATCH_SIZE, with_vectors: bool = False) -> Iterator[Record]:
        collection = self._open(create=False)
        if collection is None:
            return
        include = ["documents", "metadatas"] + (["embeddings"] if with_vectors else [])
        offset = 0
        while True:
            page = collection.get(include=include, limit=batch_size, offset=offset)
            ids = page["ids"]
            if len(ids) == 0:
                return
            vectors = page["embeddings"] if with_vectors else None
            for index, doc_id in enumerate(ids):
                yield Record(
                    id=doc_id,
                    text=page["documents"][index] or "",
                    metadata=page["metadatas"][index] or {},
                    vector=[float(x) for x in vectors[index]] if vectors is not None else None,
                )
            offset += len(ids)
            if len(ids) < batch_size:
                return

    def reset(self) -> None:
        """删除集合（不删目录里的其他文件）；已有的 collection 对象随之失效，下次使用时重新打开。"""
        with self._lock:
            if self.path.exists():
                client = self._get_client()
                try:
                    client.delete_collection(self.collection_name)
                except Exception:  # 集合本来就不存在
                    logger.debug("Chroma 集合 %s 不存在，无需删除", self.collection_name)
            self._collection = None

    def close(self) -> None:
        with self._lock:
            self._collection = None
            self._client = None
