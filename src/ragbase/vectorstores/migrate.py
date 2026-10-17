"""向量库迁移：把一个后端里的 ID、向量、正文、元数据按批复制到另一个后端，不重新向量化。

片段 ID 与正文都不变，所以 .ingest_meta.json 和 BM25 分词缓存继续有效，迁移后只需把 RAG_VECTOR_STORE 切到新后端。
"""
from collections.abc import Callable
from dataclasses import dataclass

from ragbase.vectorstores.base import VectorStore, clean_metadata

MIGRATE_BATCH_SIZE = 500


class MigrationError(RuntimeError):
    pass


@dataclass
class MigrationResult:
    source_count: int
    copied: int
    target_count: int


def migrate_store(
    source: VectorStore,
    target: VectorStore,
    batch_size: int = MIGRATE_BATCH_SIZE,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> MigrationResult:
    """source -> target。目标非空时拒绝，除非 force=True（先清空目标再复制）。完成后校验三方片段数一致。"""
    if source is target or (source.name == target.name and source.location == target.location):
        raise MigrationError("源库与目标库是同一个，无需迁移")
    source_count = source.count()
    if not source_count:
        raise MigrationError(f"源库（{source.name}：{source.location}）为空或不存在，没有可迁移的数据")
    target_count = target.count()
    if target_count:
        if not force:
            raise MigrationError(
                f"目标库（{target.name}：{target.location}）已有 {target_count} 个片段；确认要覆盖请加 --force（会先清空目标库）"
            )
        target.reset()

    copied = 0
    ids, vectors, texts, metadatas = [], [], [], []

    def flush() -> None:
        nonlocal copied
        if ids:
            target.upsert(ids, vectors, texts, metadatas)
            copied += len(ids)
            ids.clear(), vectors.clear(), texts.clear(), metadatas.clear()
            if progress:
                progress(copied, source_count)

    for record in source.iter_all(batch_size=batch_size, with_vectors=True):
        if record.vector is None:
            raise MigrationError(f"片段 {record.id} 没有向量，无法迁移")
        ids.append(record.id)
        vectors.append(record.vector)
        texts.append(record.text)
        metadatas.append(clean_metadata(record.metadata))
        if len(ids) >= batch_size:
            flush()
    flush()

    result = MigrationResult(source_count=source_count, copied=copied, target_count=target.count())
    if not (result.source_count == result.copied == result.target_count):
        raise MigrationError(
            f"迁移后片段数不一致：源库 {result.source_count}，已复制 {result.copied}，目标库 {result.target_count}"
        )
    return result
