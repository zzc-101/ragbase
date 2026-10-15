"""混合检索：向量召回 + BM25 关键词召回，用 RRF（倒数排名融合）合并。

- 向量路：直接查 Chroma collection，拿到片段 ID。
- BM25 路：启动时从 Chroma 取出全部片段文本，用 jieba 分词后建 bm25s 索引。
  分词结果缓存在 .bm25_cache/，入库后（片段数或 .ingest_meta.json 变化）自动失效。
- RRF 只看排名不看分数，所以不需要对余弦距离和 BM25 分数做归一化。
"""
import hashlib
import json
import logging
import re
import time
from pathlib import Path

import bm25s
import jieba
from langchain_core.documents import Document

from rag_common import BASE_DIR, META_FILE

TOKENIZER_VERSION = "2"  # 分词规则（停用词、过滤逻辑等）变化时递增，使缓存失效
USERDICT_FILE = BASE_DIR / "bm25_userdict.txt"  # 放仓库根目录：data/*.txt 会被当文档入库
CACHE_DIR = BASE_DIR / ".bm25_cache"
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
        if USERDICT_FILE.exists():
            _tokenizer.load_userdict(str(USERDICT_FILE))
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


def _cache_key(ids: list[str]) -> str:
    """缓存键：分词版本 + 片段数 + 入库元数据（缺失则用 ID 列表）+ 用户词典。"""
    corpus = _file_digest(META_FILE)
    if corpus == "-":
        corpus = hashlib.md5("\n".join(ids).encode()).hexdigest()
    raw = f"{TOKENIZER_VERSION}|{len(ids)}|{corpus}|{_file_digest(USERDICT_FILE)}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def tokenize_corpus(ids: list[str], texts: list[str]) -> tuple[list[list[str]], bool]:
    """分词整个语料，优先读缓存。返回 (分词结果, 是否命中缓存)。"""
    cache_file = CACHE_DIR / f"tokens-{_cache_key(ids)}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached["ids"] == ids:
                return cached["tokens"], True
        except (OSError, ValueError, KeyError):
            pass  # 缓存损坏就重建

    corpus_tokens = [tokenize(text) for text in texts]
    CACHE_DIR.mkdir(exist_ok=True)
    for old in CACHE_DIR.glob("tokens-*.json"):
        old.unlink()
    temporary = cache_file.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"ids": ids, "tokens": corpus_tokens}, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(cache_file)
    return corpus_tokens, False


class HybridRetriever:
    """向量 / BM25 / 混合三种检索模式，.invoke(question) 返回 Document 列表。

    每个返回的 Document.metadata 额外带有：
    rrf_score（融合分数）、vec_rank / bm25_rank（1 起的路内排名，未命中为 None）、chunk_id。
    """

    def __init__(
        self,
        collection,
        embeddings,
        mode: str = "hybrid",
        per_route_k: int = 30,
        final_k: int = 20,
    ):
        if mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES} 之一，收到 {mode!r}")
        self.collection = collection
        self.embeddings = embeddings
        self.mode = mode
        self.per_route_k = per_route_k
        self.final_k = final_k
        self.build_seconds = 0.0
        self.cache_hit: bool | None = None
        self._bm25 = None
        self._corpus: dict[str, tuple[str, dict]] = {}
        self._ids: list[str] = []
        if mode != "vector":
            self._build_bm25()

    def _build_bm25(self) -> None:
        """从 Chroma 取全部片段，分词（或读缓存）并建 BM25 索引。"""
        start = time.perf_counter()
        data = self.collection.get(include=["documents", "metadatas"])
        self._ids = data["ids"]
        self._corpus = {
            doc_id: (text, metadata or {})
            for doc_id, text, metadata in zip(
                data["ids"], data["documents"], data["metadatas"]
            )
        }
        corpus_tokens, self.cache_hit = tokenize_corpus(self._ids, data["documents"])
        self._bm25 = bm25s.BM25()
        self._bm25.index(corpus_tokens, show_progress=False)
        self.build_seconds = time.perf_counter() - start

    def _vector_route(self, question: str, k: int) -> list[tuple[str, str, dict]]:
        result = self.collection.query(
            query_embeddings=[self.embeddings.embed_query(question)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            (doc_id, text, metadata or {})
            for doc_id, text, metadata in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0]
            )
        ]

    def _bm25_route(self, question: str, k: int) -> list[tuple[str, str, dict]]:
        query_tokens = tokenize(question)
        if not query_tokens:
            return []
        k = min(k, len(self._ids))
        indices, scores = self._bm25.retrieve(
            [query_tokens], k=k, show_progress=False, n_threads=1
        )
        hits = []
        for index, score in zip(indices[0], scores[0]):
            if score <= 0:  # 查询词都不在语料里时，bm25s 会返回全 0 分的无关片段
                continue
            doc_id = self._ids[int(index)]
            text, metadata = self._corpus[doc_id]
            hits.append((doc_id, text, metadata))
        return hits

    def invoke(self, question: str) -> list[Document]:
        routes: list[tuple[str, list[tuple[str, str, dict]]]] = []
        if self.mode == "vector":
            routes.append(("vec", self._vector_route(question, self.final_k)))
        elif self.mode == "bm25":
            routes.append(("bm25", self._bm25_route(question, self.final_k)))
        else:
            routes.append(("vec", self._vector_route(question, self.per_route_k)))
            routes.append(("bm25", self._bm25_route(question, self.per_route_k)))

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
