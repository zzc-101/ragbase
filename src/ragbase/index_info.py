"""索引的 embedding 签名：记录“这份索引是用哪个 embedding 建的”，配置变了就拒绝继续用。

不同 embedding 模型（或同一模型的不同维度 / 不同提供方的不同版本）产生的向量互不兼容：维度不同会直接报错，
维度相同则会悄悄检索出错误结果。所以入库时把签名（提供方 / 模型 / 维度）写到索引旁边，serve / ask / chat / 增量 ingest
启动前比对，不一致就拒绝并提示 `ragbase ingest --full` 重建。

存放位置选择：<主目录>/.index_info.json（Settings.index_info_path），而不是向量库集合的元数据——
1. 与向量库后端无关：Chroma 有集合元数据，Qdrant 没有现成的自由字段，用文件两边行为一致；
2. `ragbase migrate-store` 只搬向量，不搬集合元数据，文件签名在迁移后自然仍然有效；
3. 与 .ingest_meta.json 同目录同风格，便于备份和排查。

旧索引（没有该文件）视为用默认本地模型建的：当前配置恰好是默认值就静默补写；否则拒绝。
本模块不导入 torch 等重型依赖。
"""
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ragbase.config import DEFAULT_EMBEDDING_MODEL, Settings

logger = logging.getLogger(__name__)

REBUILD_HINT = "请运行 ragbase ingest --full 重建索引（或把 embedding 配置改回建索引时的值）"


class IndexMismatchError(RuntimeError):
    """当前 embedding 配置与已有索引不一致；str(exc) 是可直接展示给用户的中文说明（不含密钥）。"""


@dataclass(frozen=True)
class EmbeddingSignature:
    """决定向量是否兼容的三项：提供方、模型、（显式配置的）维度。地址 / 密钥 / 批大小不计入。"""

    provider: str
    model: str
    dimensions: int | None = None

    def describe(self) -> str:
        dims = f"，维度 {self.dimensions}" if self.dimensions else ""
        return f"{self.provider} / {self.model}{dims}"

    def to_dict(self) -> dict:
        return {"provider": self.provider, "model": self.model, "dimensions": self.dimensions}

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingSignature":
        dimensions = data.get("dimensions")
        return cls(str(data["provider"]), str(data["model"]), int(dimensions) if dimensions else None)


# 没有签名文件的旧索引，一定是默认本地模型建的（这个功能之前只有这一种 embedding）
LEGACY_SIGNATURE = EmbeddingSignature("local", DEFAULT_EMBEDDING_MODEL, None)


def current_signature(settings: Settings) -> EmbeddingSignature:
    """当前配置的签名。本地模型的输出维度由模型决定，不看 RAG_EMBEDDING_DIMENSIONS。"""
    dimensions = settings.embedding_dimensions if settings.embedding_provider != "local" else None
    return EmbeddingSignature(settings.embedding_provider, settings.embedding_model, dimensions)


def load_index_info(path: Path) -> dict | None:
    """读取签名文件；不存在返回 None；内容损坏抛 IndexMismatchError（宁可拒绝也不要猜）。"""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        EmbeddingSignature.from_dict(data["embedding"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise IndexMismatchError(
            f"索引信息文件 {path} 无法读取（{type(exc).__name__}）。{REBUILD_HINT}；确认无误后也可删除该文件"
        ) from exc
    return data


def save_index_info(path: Path, signature: EmbeddingSignature, dimension: int | None = None) -> None:
    """原子写入签名；dimension 是实际向量长度（已知时记录，用于入库时的长度校验）。"""
    path = Path(path)
    old = None
    try:
        old = load_index_info(path)
    except IndexMismatchError:
        pass
    if dimension is None and old and old["embedding"] == signature.to_dict():
        dimension = old.get("dimension")  # 签名没变就保留已知的实际维度
    data = {
        "embedding": signature.to_dict(),
        "dimension": dimension,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _mismatch_message(stored: EmbeddingSignature, current: EmbeddingSignature, legacy: bool) -> str:
    origin = "旧索引没有 embedding 记录，视为用默认本地模型建立" if legacy else "索引建立时使用"
    return (
        f"当前 embedding 配置与索引不一致，请运行 ragbase ingest --full 重建索引。\n"
        f"  {origin}：{stored.describe()}\n"
        f"  当前配置：{current.describe()}\n"
        f"（不同 embedding 的向量互不兼容，继续使用会得到错误的检索结果；若想继续用原索引，请把 embedding 配置改回上面的值。）"
    )


@dataclass(frozen=True)
class IndexCheck:
    """检查结果。status：empty（索引为空，无约束）/ match / legacy_default（旧索引 + 默认配置，兼容）/ mismatch。"""

    status: str
    stored: EmbeddingSignature | None = None
    current: EmbeddingSignature | None = None
    message: str = ""


def check_index(settings: Settings, store, write_legacy: bool = False) -> IndexCheck:
    """比对当前 embedding 配置与索引签名（只读；write_legacy=True 时旧索引 + 默认配置会补写签名文件）。

    store 是向量库对象（只用 exists() / count()）。索引为空不受约束：随后的入库会（重新）写入签名。
    签名文件损坏时抛 IndexMismatchError。
    """
    current = current_signature(settings)
    if not (store.exists() and store.count()):
        return IndexCheck("empty", None, current)
    info = load_index_info(settings.index_info_path)
    if info is None:
        if current == LEGACY_SIGNATURE:
            if write_legacy:
                try:
                    save_index_info(settings.index_info_path, current)
                except OSError:
                    logger.warning("无法写入索引信息文件 %s（目录不可写？），已跳过", settings.index_info_path)
            return IndexCheck("legacy_default", LEGACY_SIGNATURE, current)
        return IndexCheck("mismatch", LEGACY_SIGNATURE, current, _mismatch_message(LEGACY_SIGNATURE, current, legacy=True))
    stored = EmbeddingSignature.from_dict(info["embedding"])
    if stored == current:
        return IndexCheck("match", stored, current)
    return IndexCheck("mismatch", stored, current, _mismatch_message(stored, current, legacy=False))


def ensure_index_compatible(settings: Settings, store) -> IndexCheck:
    """serve / ask / chat / 增量 ingest 启动前调用：不一致抛 IndexMismatchError；旧索引 + 默认配置静默补写签名文件。"""
    result = check_index(settings, store, write_legacy=True)
    if result.status == "mismatch":
        raise IndexMismatchError(result.message)
    return result


def store_dimension(store) -> int | None:
    """索引里已有向量的长度（读第一条记录）；索引为空或后端不支持返回 None。"""
    try:
        for record in store.iter_all(batch_size=1, with_vectors=True):
            return len(record.vector) if record.vector is not None else None
    except Exception:
        logger.debug("读取索引向量维度失败", exc_info=True)
    return None


class IndexRecorder:
    """入库时的两个钩子：begin() 在第一次写向量之前登记签名；wrap(embed_fn) 校验向量长度并记录实际维度。

    用法：guard = IndexRecorder(settings, store); run_ingest(..., guard.wrap(embed_fn), ..., before_write=guard.begin)。
    ragbase ingest --full 先清空索引再入库，begin() 会覆盖旧签名；增量入库时签名已由 ensure_index_compatible 校验过。
    """

    def __init__(self, settings: Settings, store):
        self.settings = settings
        self.store = store
        self.signature = current_signature(settings)
        self._checked = False  # 第一批向量的长度是否已校验并记录

    def begin(self) -> None:
        save_index_info(self.settings.index_info_path, self.signature)

    def wrap(self, embed_fn: Callable) -> Callable:
        def embed(texts):
            vectors = embed_fn(texts)
            if vectors:
                self._check(len(vectors[0]))
            return vectors

        return embed

    def _check(self, length: int) -> None:
        if self._checked:
            return
        expected = store_dimension(self.store)
        if expected is not None and expected != length:
            raise IndexMismatchError(
                f"embedding 返回的向量长度 {length} 与索引中已有向量的长度 {expected} 不一致，"
                f"无法混合写入。{REBUILD_HINT}"
            )
        self._checked = True
        save_index_info(self.settings.index_info_path, self.signature, dimension=length)
