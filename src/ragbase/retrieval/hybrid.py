"""混合检索：向量召回 + BM25 关键词召回，用 RRF（倒数排名融合）合并。

- 向量路：直接查 Chroma collection，拿到片段 ID。
- BM25 路：启动时从 Chroma 取出全部片段文本，用 jieba 分词后建 bm25s 索引。
  分词输入 = 章节路径（heading_path，若有）+ 正文。分词结果缓存在 .bm25_cache/，
  缓存键包含全部片段的内容摘要：入库 / 删除 / 章节标题变化后自动失效，旧缓存文件会被清掉。
- 索引快照：BM25 索引、语料和“书目录”打包成不可变的 _Snapshot，整体替换一个引用即可原子切换。
  入库完成后 refresh() 先建好新快照再替换，正在进行的检索始终看到完整的旧快照或新快照。
- 按书过滤（sources）：向量路用 Chroma 的 where={"source": {"$in": [...]}}，BM25 路用 bm25s 的 weight_mask。
- RRF 只看排名不看分数，所以不需要对余弦距离和 BM25 分数做归一化。
"""
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import bm25s
import jieba
import numpy as np
from langchain_core.documents import Document

from ragbase.config import get_settings, userdict_file
from ragbase.ingest.common import bm25_input

logger = logging.getLogger(__name__)

TOKENIZER_VERSION = "3"  # 分词规则（停用词、过滤逻辑、分词输入含章节路径等）变化时递增，使缓存失效
RRF_K = 60

MODES = ("vector", "bm25", "hybrid")

# 只放高频虚词和疑问词；BM25 的 IDF 本身会压低常见词，不需要大停用词表。
STOPWORDS = frozenset(
    "的 了 是 在 和 与 及 或 吗 呢 吧 啊 什么 如何 哪些 怎么 怎样 为什么 是否 "
    "有 对 等 并 把 被 从 到 之 其 这 那 个 将 可以 以及 一个 进行".split()
)
_TOKEN_CHAR = re.compile(r"[0-9a-z一-鿿]")
# PDF 抽取常把缩写拆成单字母（"T O G A F"），合并回 "togaf" 才能被检索到
_SPACED_LETTERS = re.compile(r"\b(?:[a-z] ){2,}[a-z]\b")

jieba.setLogLevel(logging.WARNING)
_tokenizer: jieba.Tokenizer | None = None


def _get_tokenizer() -> jieba.Tokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = jieba.Tokenizer()
        userdict = userdict_file()  # 项目主目录（RAG_HOME）下的 bm25_userdict.txt；放在 data/ 下会被当文档入库
        if userdict.exists():
            _tokenizer.load_userdict(str(userdict))
    return _tokenizer


def tokenize(text: str) -> list[str]:
    """索引和查询共用的分词：jieba 搜索模式 + 小写 + 去标点/停用词。

    英文单词、数字、缩写（TOGAF、ISO 9001）按原样保留为小写 token；
    单个 ASCII 字符（OCR 噪声、选项 A/B/C）丢弃；
    被空格拆开的 3 个以上单字母（"T O G A F"）先合并再分词。
    """
    text = _SPACED_LETTERS.sub(lambda m: m.group().replace(" ", ""), text.lower())
    tokens = []
    for token in _get_tokenizer().cut_for_search(text):
        token = token.strip()
        if not token or not _TOKEN_CHAR.search(token) or token in STOPWORDS:
            continue
        if len(token) == 1 and token.isascii():
            continue
        tokens.append(token)
    return tokens


def rrf_fuse(
    ranked_id_lists: list[list[str]],
    k: int = RRF_K,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """倒数排名融合：score(id) = Σ weight / (k + rank)，rank 从 1 开始。

    返回按融合分数从高到低排序的 (id, score)；分数相同时保持先出现者在前。
    """
    weights = weights or [1.0] * len(ranked_id_lists)
    scores: dict[str, float] = {}
    for ids, weight in zip(ranked_id_lists, weights):
        for rank, doc_id in enumerate(ids, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _file_digest(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else "-"


def _cache_key(ids: list[str], texts: list[str]) -> str:
    """缓存键：分词版本 + 片段数 + 全部片段（ID + 分词输入）的内容摘要 + 用户词典。

    用内容摘要而不是入库元数据文件：不管片段是怎么改的（入库、删除、章节标题变化），
    只要分词输入变了键就变，不会读到过期缓存。
    """
    digest = hashlib.md5()
    for doc_id, text in zip(ids, texts):
        digest.update(doc_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\1")
    raw = f"{TOKENIZER_VERSION}|{len(ids)}|{digest.hexdigest()}|{_file_digest(userdict_file())}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def tokenize_corpus(
    ids: list[str], texts: list[str], cache_dir: Path | None = None
) -> tuple[list[list[str]], bool]:
    """分词整个语料（texts 是已拼好章节路径的分词输入），优先读缓存。返回 (分词结果, 是否命中缓存)。

    写入新缓存时清掉目录里所有旧的 tokens-*.json 和残留的 .tmp，所以缓存目录里始终最多一份有效缓存。
    """
    cache_dir = Path(cache_dir or get_settings().cache_dir)  # 缺省为配置里的 cache_dir（.bm25_cache/）
    cache_file = cache_dir / f"tokens-{_cache_key(ids, texts)}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached["ids"] == ids:
                return cached["tokens"], True
        except (OSError, ValueError, KeyError):
            pass  # 缓存损坏就重建

    corpus_tokens = [tokenize(text) for text in texts]
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in [*cache_dir.glob("tokens-*.json"), *cache_dir.glob("tokens-*.tmp")]:
        old.unlink(missing_ok=True)
    temporary = cache_file.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"ids": ids, "tokens": corpus_tokens}, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(cache_file)
    return corpus_tokens, False


class UnknownSourceError(ValueError):
    """sources 里有索引中不存在的书名；unknown 是不存在的名字，valid 是全部可选书名。"""

    def __init__(self, unknown: list[str], valid: list[str]):
        super().__init__(f"未知的书名：{unknown}；可选：{valid}")
        self.unknown = unknown
        self.valid = valid


@dataclass(frozen=True)
class _Snapshot:
    """某一时刻的索引快照。构建后不再修改；换索引就是整体替换 HybridRetriever._snapshot 这一个引用。"""

    ids: list[str]
    corpus: dict[str, tuple[str, dict]]  # chunk_id -> (正文, 元数据)
    bm25: object | None  # bm25s.BM25；未构建 BM25 时为 None
    built_at: float  # time.time()
    build_seconds: float
    cache_hit: bool | None
    sources: dict[str, tuple[str, ...]]  # 书名（source 的 basename）-> 元数据里 source 的原始取值（旧索引是绝对路径）
    positions: dict[str, np.ndarray]  # source 原始取值 -> 其片段在 ids 里的下标
    catalog: tuple[dict, ...]  # 每本书的统计：name / chunk_count / heading_chunks / has_headings


def _build_snapshot(collection, with_bm25: bool, cache_dir: Path | None) -> _Snapshot:
    """从 Chroma 取全部片段，建语料、书目录，并按需分词（或读缓存）建 BM25 索引。"""
    start = time.perf_counter()
    data = collection.get(include=["documents", "metadatas"])
    ids = list(data["ids"])
    documents = list(data["documents"])
    metadatas = [metadata or {} for metadata in data["metadatas"]]
    corpus = {doc_id: (text, metadata) for doc_id, text, metadata in zip(ids, documents, metadatas)}

    source_values: dict[str, dict[str, None]] = {}
    positions: dict[str, list[int]] = {}
    stats: dict[str, dict] = {}
    for position, metadata in enumerate(metadatas):
        source = metadata.get("source")
        if not isinstance(source, str) or not source:
            continue
        name = os.path.basename(source)
        source_values.setdefault(name, {})[source] = None
        positions.setdefault(source, []).append(position)
        entry = stats.setdefault(name, {"name": name, "chunk_count": 0, "heading_chunks": 0})
        entry["chunk_count"] += 1
        if metadata.get("heading_path"):
            entry["heading_chunks"] += 1
    catalog = tuple(
        {**entry, "has_headings": entry["heading_chunks"] > 0}
        for _, entry in sorted(stats.items())
    )

    bm25, cache_hit = None, None
    if with_bm25 and ids:
        index_texts = [bm25_input(text, metadata) for text, metadata in zip(documents, metadatas)]
        corpus_tokens, cache_hit = tokenize_corpus(ids, index_texts, cache_dir)
        bm25 = bm25s.BM25()
        bm25.index(corpus_tokens, show_progress=False)
    seconds = time.perf_counter() - start
    logger.info(
        "索引快照就绪：%d 个片段，%d 本书，BM25 %s，用时 %.1fs%s",
        len(ids),
        len(catalog),
        "已构建" if bm25 is not None else "未构建",
        seconds,
        "" if cache_hit is None else f"（分词缓存{'命中' if cache_hit else '未命中'}）",
    )
    return _Snapshot(
        ids=ids,
        corpus=corpus,
        bm25=bm25,
        built_at=time.time(),
        build_seconds=seconds,
        cache_hit=cache_hit,
        sources={name: tuple(values) for name, values in source_values.items()},
        positions={source: np.asarray(indexes, dtype=np.int64) for source, indexes in positions.items()},
        catalog=catalog,
    )


class HybridRetriever:
    """向量 / BM25 / 混合三种检索模式，.invoke(question, mode=None, sources=None) 返回 Document 列表。

    构造时的 mode 是默认模式，invoke 可按次覆盖；用到 BM25 的模式要求索引已构建
    （mode != "vector" 或 build_bm25=True 时构建，否则 invoke 会抛出明确错误）。
    sources 是书名（文件 basename）列表，只在这些书里检索；书名必须是 catalog() 里列出的，否则抛 UnknownSourceError。

    每个返回的 Document.metadata 额外带有：
    rrf_score（融合分数）、vec_rank / bm25_rank（1 起的路内排名，未命中为 None）、chunk_id。

    入库后调用 refresh() 重建快照；可从任意线程调用，调用期间检索照常使用旧快照。
    """

    def __init__(
        self,
        collection,
        embeddings,
        mode: str = "hybrid",
        per_route_k: int = 30,
        final_k: int = 20,
        build_bm25: bool | None = None,
        cache_dir: Path | None = None,
    ):
        if mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES} 之一，收到 {mode!r}")
        self.collection = collection
        self.embeddings = embeddings
        self.mode = mode
        self.per_route_k = per_route_k
        self.final_k = final_k
        self.cache_dir = cache_dir
        self._snapshot: _Snapshot | None = None
        self._build_lock = threading.Lock()  # 同一时刻只建一个快照
        if build_bm25 is None:
            build_bm25 = mode != "vector"
        self._with_bm25 = build_bm25
        if build_bm25:
            self._snapshot = _build_snapshot(collection, True, cache_dir)

    @property
    def bm25_ready(self) -> bool:
        snapshot = self._snapshot
        return snapshot is not None and snapshot.bm25 is not None

    @property
    def bm25_built_at(self) -> float | None:
        """当前 BM25 索引的构建时间（Unix 时间戳）；未构建为 None。"""
        snapshot = self._snapshot
        return snapshot.built_at if snapshot is not None and snapshot.bm25 is not None else None

    @property
    def bm25_chunk_count(self) -> int | None:
        """当前 BM25 索引覆盖的片段数；未构建为 None。"""
        snapshot = self._snapshot
        return len(snapshot.ids) if snapshot is not None and snapshot.bm25 is not None else None

    @property
    def build_seconds(self) -> float:
        return self._snapshot.build_seconds if self._snapshot is not None else 0.0

    @property
    def cache_hit(self) -> bool | None:
        return self._snapshot.cache_hit if self._snapshot is not None else None

    def refresh(self) -> None:
        """重新读取 Chroma 并建新快照，建好后原子替换；失败时保留旧快照并抛出异常。"""
        with self._build_lock:
            new_snapshot = _build_snapshot(self.collection, self._with_bm25, self.cache_dir)
            self._snapshot = new_snapshot  # 单个引用赋值：读者要么看到旧快照，要么看到新快照

    def _catalog_snapshot(self) -> _Snapshot:
        """取快照；纯向量检索器（没建 BM25）第一次需要书目录时才惰性建一份只含语料的快照。"""
        snapshot = self._snapshot
        if snapshot is None:
            with self._build_lock:
                if self._snapshot is None:
                    self._snapshot = _build_snapshot(self.collection, self._with_bm25, self.cache_dir)
                snapshot = self._snapshot
        return snapshot

    def catalog(self) -> list[dict]:
        """索引中的书：[{name, chunk_count, heading_chunks, has_headings}]，按书名排序。"""
        return [dict(entry) for entry in self._catalog_snapshot().catalog]

    def resolve_sources(self, names: list[str]) -> list[str]:
        """书名（basename）-> 元数据里 source 的原始取值（旧索引是绝对路径，所以不能自己拼）。"""
        return self._resolve(self._catalog_snapshot(), names)

    @staticmethod
    def _resolve(snapshot: _Snapshot, names: list[str]) -> list[str]:
        unknown = [name for name in dict.fromkeys(names) if name not in snapshot.sources]
        if unknown:
            raise UnknownSourceError(unknown, sorted(snapshot.sources))
        values: dict[str, None] = {}
        for name in names:
            values.update(dict.fromkeys(snapshot.sources[name]))
        return list(values)

    def _vector_route(self, question: str, k: int, allowed: list[str] | None) -> list[tuple[str, str, dict]]:
        options = {} if allowed is None else {"where": {"source": {"$in": allowed}}}
        result = self.collection.query(
            query_embeddings=[self.embeddings.embed_query(question)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
            **options,
        )
        return [
            (doc_id, text, metadata or {})
            for doc_id, text, metadata in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0]
            )
        ]

    @staticmethod
    def _bm25_route(
        snapshot: _Snapshot, question: str, k: int, allowed: list[str] | None
    ) -> list[tuple[str, str, dict]]:
        query_tokens = tokenize(question)
        if not query_tokens:
            return []
        weight_mask = None
        if allowed is None:
            k = min(k, len(snapshot.ids))
        else:
            # 0/1 掩码：不在所选书里的片段得分被乘成 0，下面的 score <= 0 过滤会把它们去掉
            weight_mask = np.zeros(len(snapshot.ids), dtype=np.float32)
            for source in allowed:
                weight_mask[snapshot.positions[source]] = 1.0
            k = min(k, int(weight_mask.sum()))  # k 不能超过过滤后的语料大小
        if k <= 0:
            return []
        indices, scores = snapshot.bm25.retrieve(
            [query_tokens], k=k, show_progress=False, n_threads=1, weight_mask=weight_mask
        )
        hits = []
        for index, score in zip(indices[0], scores[0]):
            if score <= 0:  # 查询词都不在（过滤后的）语料里时，bm25s 会返回全 0 分的无关片段
                continue
            doc_id = snapshot.ids[int(index)]
            text, metadata = snapshot.corpus[doc_id]
            hits.append((doc_id, text, metadata))
        return hits

    def invoke(self, question: str, mode: str | None = None, sources: list[str] | None = None) -> list[Document]:
        mode = mode or self.mode
        if mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES} 之一，收到 {mode!r}")
        snapshot = self._snapshot  # 只取一次：整个调用用同一份快照，即使中途被 refresh 替换也不受影响
        if mode != "vector" and (snapshot is None or snapshot.bm25 is None):
            raise RuntimeError(f"mode={mode!r} 需要 BM25 索引，但构造时未构建（请用 build_bm25=True）")
        allowed = None
        if sources:
            snapshot = snapshot or self._catalog_snapshot()
            allowed = self._resolve(snapshot, sources)
        routes: list[tuple[str, list[tuple[str, str, dict]]]] = []
        if mode == "vector":
            routes.append(("vec", self._vector_route(question, self.final_k, allowed)))
        elif mode == "bm25":
            routes.append(("bm25", self._bm25_route(snapshot, question, self.final_k, allowed)))
        else:
            routes.append(("vec", self._vector_route(question, self.per_route_k, allowed)))
            routes.append(("bm25", self._bm25_route(snapshot, question, self.per_route_k, allowed)))

        ranks: dict[str, dict[str, int]] = {"vec": {}, "bm25": {}}
        content: dict[str, tuple[str, dict]] = {}
        for name, hits in routes:
            for rank, (doc_id, text, metadata) in enumerate(hits, 1):
                ranks[name][doc_id] = rank
                content.setdefault(doc_id, (text, metadata))

        fused = rrf_fuse([[hit[0] for hit in hits] for _, hits in routes])
        documents = []
        for doc_id, score in fused[: self.final_k]:
            text, metadata = content[doc_id]
            documents.append(Document(
                page_content=text,
                metadata={
                    **metadata,
                    "chunk_id": doc_id,
                    "rrf_score": score,
                    "vec_rank": ranks["vec"].get(doc_id),
                    "bm25_rank": ranks["bm25"].get(doc_id),
                },
            ))
        return documents
