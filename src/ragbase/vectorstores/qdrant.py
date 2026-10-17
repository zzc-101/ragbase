"""Qdrant 后端：本地嵌入式模式（目录）或服务端模式（URL）。

这是整个项目里唯一 import qdrant_client 的地方。

- 点 ID：Qdrant 只接受 UUID / 整数，所以字符串片段 ID 用 uuid5(固定命名空间, chunk_id) 映射，
  原始 ID 存在 payload["chunk_id"]，读出时还原；同一个 chunk_id 永远映射到同一个点（upsert 覆盖）。
- payload：{"chunk_id": 原始ID, "document": 正文, "metadata": {标量元数据}}。元数据嵌套在 "metadata" 下，
  不会和 chunk_id / document 重名。fid 与 source 建 keyword 索引（metadata.fid / metadata.source；本地模式不支持 payload 索引，会跳过）。
- 距离：余弦（Chroma 的 L2 对归一化向量排序等价）。向量维度取自第一次 upsert，集合在那时才创建。
- 过滤：fid 用 MatchValue，source 用 MatchAny；翻页用 scroll。
- 本地模式一个目录同一时刻只允许一个 QdrantClient（目录锁）：本模块按目录 / URL 在进程内共享同一个 client，
  所以同一进程里多次创建 QdrantStore 是安全的；但不能同时运行两个进程（如 ragbase serve 与 ragbase ingest）
  使用同一个本地目录——第二个会报“already accessed by another instance”。需要多进程访问请用服务端模式。
  本地模式的 client 不是线程安全的，这里用锁串行化所有调用（服务端模式不加锁）。
"""
import threading
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from ragbase.vectorstores.base import ITER_BATCH_SIZE, Hit, Record

POINT_NAMESPACE = uuid.UUID("6f1d2c1e-8b0a-5c55-9a52-5f7a1c0de001")  # 固定命名空间：不能改，否则同一个 chunk_id 会映射成另一个点
INDEXED_FIELDS = ("fid", "source")

_registry_lock = threading.Lock()
_clients: dict[str, tuple[QdrantClient, threading.RLock]] = {}


def point_id(chunk_id: str) -> str:
    """字符串片段 ID -> 稳定的 UUID 点 ID。"""
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def _client_key(path: Path | None, url: str | None) -> str:
    return f"url:{url}" if url else f"path:{Path(path).expanduser().resolve()}"


def _acquire(path: Path | None, url: str | None, api_key: str | None) -> tuple[QdrantClient, threading.RLock]:
    """按目录 / URL 取进程内共享的 client（没有就创建）及其锁。"""
    key = _client_key(path, url)
    with _registry_lock:
        entry = _clients.get(key)
        if entry is None:
            if url:
                client = QdrantClient(url=url, api_key=api_key)
            else:
                Path(path).expanduser().resolve().mkdir(parents=True, exist_ok=True)
                client = QdrantClient(path=str(Path(path).expanduser().resolve()))
            entry = _clients[key] = (client, threading.RLock())
        return entry


def _release(path: Path | None, url: str | None) -> None:
    with _registry_lock:
        entry = _clients.pop(_client_key(path, url), None)
    if entry is not None:
        with entry[1]:
            entry[0].close()


class QdrantStore:
    name = "qdrant"

    def __init__(
        self,
        collection: str = "rag",
        path: str | Path | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ):
        if not url and path is None:
            raise ValueError("QdrantStore 需要 path（本地模式）或 url（服务端模式）")
        self.collection_name = collection
        self.path = Path(path) if path is not None and not url else None
        self.url = url or None
        self._api_key = api_key
        self.location = self.url or str(self.path)
        self._local = not self.url  # 本地模式：需要串行化，且不支持 payload 索引

    # ---- 内部 -------------------------------------------------------------------------------

    def _call(self, func):
        """取共享 client 并调用 func(client)；本地模式下持锁串行化。"""
        client, lock = _acquire(self.path, self.url, self._api_key)
        if not self._local:
            return func(client)
        with lock:
            return func(client)

    def _path_missing(self) -> bool:
        """本地模式且目录还不存在：只读探测时直接按“空库”处理，不去创建目录。"""
        return self.path is not None and not self.path.exists()

    def _ensure_collection(self, client: QdrantClient, dim: int) -> None:
        if client.collection_exists(self.collection_name):
            return
        client.create_collection(
            self.collection_name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        if self._local:  # 本地模式不支持 payload 索引（过滤是内存里逐条匹配），建了只会报警告
            return
        for field_name in INDEXED_FIELDS:
            client.create_payload_index(
                self.collection_name, f"metadata.{field_name}", field_schema=models.PayloadSchemaType.KEYWORD
            )

    def _has(self, client: QdrantClient) -> bool:
        return client.collection_exists(self.collection_name)

    @staticmethod
    def _fid_filter(fid: str) -> models.Filter:
        return models.Filter(must=[models.FieldCondition(key="metadata.fid", match=models.MatchValue(value=fid))])

    @staticmethod
    def _source_filter(sources: Sequence[str]) -> models.Filter:
        return models.Filter(
            must=[models.FieldCondition(key="metadata.source", match=models.MatchAny(any=list(sources)))]
        )

    def _scroll(self, client, scroll_filter, batch_size, with_payload, with_vectors) -> Iterator[list]:
        """逐页产出 scroll 结果（每页一个 Record 列表）。"""
        offset = None
        while True:
            points, offset = client.scroll(
                self.collection_name,
                scroll_filter=scroll_filter,
                limit=batch_size,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
            if points:
                yield points
            if offset is None:
                return

    # ---- VectorStore ------------------------------------------------------------------------

    def exists(self) -> bool:
        if self._path_missing():
            return False
        return self._call(self._has)

    def count(self) -> int:
        if self._path_missing():
            return 0

        def run(client):
            if not self._has(client):
                return 0
            return client.count(self.collection_name, exact=True).count

        return self._call(run)

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        if not ids:
            return
        vectors = np.asarray(embeddings, dtype=np.float32)
        points = [
            models.PointStruct(
                id=point_id(chunk_id),
                vector=vector.tolist(),
                payload={"chunk_id": chunk_id, "document": text, "metadata": dict(metadata or {})},
            )
            for chunk_id, vector, text, metadata in zip(ids, vectors, documents, metadatas)
        ]

        def run(client):
            self._ensure_collection(client, vectors.shape[1])
            client.upsert(self.collection_name, points=points, wait=True)

        self._call(run)

    def delete(self, ids: Sequence[str]) -> None:
        ids = list(ids)
        if not ids or self._path_missing():
            return

        def run(client):
            if self._has(client):
                client.delete(
                    self.collection_name,
                    points_selector=models.PointIdsList(points=[point_id(chunk_id) for chunk_id in ids]),
                    wait=True,
                )

        self._call(run)

    def _ids(self, scroll_filter) -> list[str]:
        if self._path_missing():
            return []

        def run(client):
            if not self._has(client):
                return []
            ids: dict[str, None] = {}
            for page in self._scroll(client, scroll_filter, ITER_BATCH_SIZE, ["chunk_id"], False):
                ids.update(dict.fromkeys(point.payload["chunk_id"] for point in page))
            return list(ids)

        return self._call(run)

    def ids_where(self, fid: str | None = None, sources: Sequence[str] | None = None) -> list[str]:
        ids: dict[str, None] = {}
        if fid:
            ids.update(dict.fromkeys(self._ids(self._fid_filter(fid))))
        if sources:
            ids.update(dict.fromkeys(self._ids(self._source_filter(sources))))
        return list(ids)

    def all_ids(self) -> list[str]:
        return self._ids(None)

    def query(self, embedding: Sequence[float], k: int, sources: Sequence[str] | None = None) -> list[Hit]:
        if k <= 0 or self._path_missing():
            return []
        query_filter = None
        if sources is not None:
            if not sources:
                return []
            query_filter = self._source_filter(sources)
        vector = np.asarray(embedding, dtype=np.float32).tolist()

        def run(client):
            if not self._has(client):
                return []
            return client.query_points(
                self.collection_name, query=vector, limit=k, query_filter=query_filter, with_payload=True
            ).points

        return [
            Hit(
                id=point.payload["chunk_id"],
                text=point.payload.get("document", ""),
                metadata=point.payload.get("metadata") or {},
                score=float(point.score),
            )
            for point in self._call(run)
        ]

    def iter_all(self, batch_size: int = ITER_BATCH_SIZE, with_vectors: bool = False) -> Iterator[Record]:
        """分页遍历；每翻一页单独加锁，翻页之间不占着锁。"""
        if self._path_missing():
            return
        offset = None
        while True:

            def run(client, offset=offset):
                if not self._has(client):
                    return [], None
                return client.scroll(
                    self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=with_vectors,
                )

            points, offset = self._call(run)
            for point in points:
                payload = point.payload or {}
                yield Record(
                    id=payload["chunk_id"],
                    text=payload.get("document", ""),
                    metadata=payload.get("metadata") or {},
                    vector=[float(x) for x in point.vector] if with_vectors else None,
                )
            if offset is None:
                return

    def reset(self) -> None:
        if self._path_missing():
            return

        def run(client):
            if self._has(client):
                client.delete_collection(self.collection_name)

        self._call(run)

    def close(self) -> None:
        """关闭并从进程内注册表移除 client（释放本地目录锁）；之后再使用本对象会自动重新打开。"""
        _release(self.path, self.url)
