"""向量库抽象：调用方（入库、检索、API、迁移）只依赖这里的 VectorStore，不直接碰 Chroma / Qdrant。

约定：
- 片段 ID 是字符串；元数据值只能是标量（str / int / float / bool），用 clean_metadata 整理。
- 集合不存在（从未写入）时，读操作按“空库”处理（count()=0、query()=[]、iter_all() 为空），exists() 为 False；
  第一次 upsert 时自动创建。
- 过滤：sources 是元数据 source 字段的原始取值列表；None 表示不过滤，空列表表示“什么都不匹配”。
- 实现要能被入库线程和请求线程同时使用。
- upsert 同 ID 覆盖：Qdrant 整体替换元数据，Chroma 按键合并（旧键保留）。入库每次写完整元数据，所以两者等价。
"""
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

ITER_BATCH_SIZE = 1000  # iter_all 默认每页条数


@dataclass(frozen=True)
class Hit:
    """向量检索的一条命中。

    query() 返回的列表已按“最相似在前”排好序，调用方只应使用这个顺序（混合检索的 RRF 只看排名）。
    score 是后端的原始数值，两个后端的含义不同、不可互相比较：
    Chroma 是距离（升序，越小越近）；Qdrant 是余弦相似度（降序，越大越近）。
    """

    id: str
    text: str
    metadata: dict
    score: float


@dataclass(frozen=True)
class Record:
    """iter_all 产出的一条完整记录；with_vectors=True 时 vector 才有值。"""

    id: str
    text: str
    metadata: dict
    vector: list[float] | None = field(default=None, repr=False)


@runtime_checkable
class VectorStore(Protocol):
    name: str  # 后端名："chroma" / "qdrant"
    location: str  # 数据位置（目录或 URL），用于提示信息

    def exists(self) -> bool:
        """集合是否已存在（不会创建任何东西）。"""

    def count(self) -> int:
        """片段总数；集合不存在为 0。"""

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        """写入片段（预先算好的向量）；同 ID 覆盖。集合不存在时自动创建。"""

    def delete(self, ids: Sequence[str]) -> None:
        """按 ID 删除；不存在的 ID 忽略。"""

    def ids_where(self, fid: str | None = None, sources: Sequence[str] | None = None) -> list[str]:
        """满足任一条件的片段 ID（去重）：元数据 fid == fid，或 source ∈ sources。两个条件都没给（None / 空）返回 []。"""

    def all_ids(self) -> list[str]:
        """全部片段 ID。"""

    def query(self, embedding: Sequence[float], k: int, sources: Sequence[str] | None = None) -> list[Hit]:
        """向量检索，返回至多 k 条，最相似的在前；sources 限定元数据 source 的取值。"""

    def iter_all(self, batch_size: int = ITER_BATCH_SIZE, with_vectors: bool = False) -> Iterator[Record]:
        """分页遍历全部片段（用于 BM25 快照、书目录、迁移）。顺序由后端决定，调用方不要依赖。"""

    def reset(self) -> None:
        """清空全部数据（删除集合）。之后 exists() 为 False，下一次 upsert 重新创建。"""

    def close(self) -> None:
        """释放资源（本地 Qdrant 会释放目录锁）。关闭后可以重新使用同一个对象（会自动重新打开）。"""


def clean_metadata(metadata: dict) -> dict:
    """向量库元数据只接受 str / int / float / bool：去掉 None，其他类型转成字符串。"""
    return {
        key: value if isinstance(value, (str, int, float, bool)) else str(value)
        for key, value in metadata.items()
        if value is not None
    }
