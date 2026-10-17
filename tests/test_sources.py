"""HybridRetriever 的按书过滤（sources）、书目录、BM25 快照刷新与分词缓存的单元测试
（假 collection / 临时 Chroma，不加载模型、不访问网络）。

运行：uv run python tests/test_sources.py
"""
import tempfile
import threading
from pathlib import Path

import chromadb

from ragbase.retrieval.hybrid import HybridRetriever, UnknownSourceError

# 旧版索引风格：source 是绝对路径，没有 heading_path / fid
LEGACY = [
    ("d1", "关键路径法计算总浮动时间", {"source": "/Users/x/data/甲书.pdf", "page": 0}),
    ("d2", "挣值分析与进度偏差", {"source": "/Users/x/data/甲书.pdf", "page": 1}),
    ("d3", "关键路径法与关键链法的比较", {"source": "/Users/x/data/乙书.pdf", "page": 4}),
    ("d4", "红烧肉的做法", {"source": "/Users/x/data/丙书.md"}),
    ("d5", "关键路径上的活动没有浮动时间", {"source": "/other/dir/乙书.pdf", "page": 9}),  # 不同目录的同名书
]


class _Collection:
    """极简 Chroma 假对象：get 返回全部片段；query 按固定顺序（逆序）返回，并支持 source 的 $in 过滤。"""

    def __init__(self, rows=LEGACY):
        self.rows = list(rows)
        self.query_kwargs = []

    def get(self, include=None):
        return {
            "ids": [row[0] for row in self.rows],
            "documents": [row[1] for row in self.rows],
            "metadatas": [dict(row[2]) for row in self.rows],
        }

    def query(self, query_embeddings, n_results, include=None, where=None):
        self.query_kwargs.append(where)
        rows = list(reversed(self.rows))
        if where is not None:
            allowed = set(where["source"]["$in"])
            rows = [row for row in rows if row[2].get("source") in allowed]
        rows = rows[:n_results]
        return {
            "ids": [[row[0] for row in rows]],
            "documents": [[row[1] for row in rows]],
            "metadatas": [[dict(row[2]) for row in rows]],
        }


class _Embeddings:
    def __init__(self, vector=(0.0,)):
        self.vector = list(vector)

    def embed_query(self, text):
        return self.vector


def _retriever(collection=None, cache_dir=None, embeddings=None, **kwargs):
    kwargs.setdefault("mode", "hybrid")
    return HybridRetriever(collection or _Collection(), embeddings or _Embeddings(), cache_dir=cache_dir, **kwargs)


def _ids(docs):
    return [doc.metadata["chunk_id"] for doc in docs]


def test_catalog_groups_by_basename_and_works_on_legacy_index():
    with tempfile.TemporaryDirectory() as tmp:
        retriever = _retriever(cache_dir=Path(tmp))
        catalog = retriever.catalog()
        assert [entry["name"] for entry in catalog] == ["丙书.md", "乙书.pdf", "甲书.pdf"]
        by_name = {entry["name"]: entry for entry in catalog}
        assert by_name["乙书.pdf"]["chunk_count"] == 2  # 两个目录下的同名文件按 basename 合并
        assert by_name["甲书.pdf"] == {"name": "甲书.pdf", "chunk_count": 2, "heading_chunks": 0, "has_headings": False}
        retriever.catalog()[0]["chunk_count"] = 999  # 返回的是副本
        assert retriever.catalog()[0]["chunk_count"] == 1


def test_catalog_counts_heading_coverage():
    rows = [
        ("a1", "正文一", {"source": "/d/书.pdf", "heading_path": "第1章 > 1.1 节"}),
        ("a2", "正文二", {"source": "/d/书.pdf"}),
        ("a3", "正文三", {"source": "/d/书.pdf", "heading_path": ""}),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        entry = _retriever(_Collection(rows), cache_dir=Path(tmp)).catalog()[0]
    assert (entry["chunk_count"], entry["heading_chunks"], entry["has_headings"]) == (3, 1, True)


def test_resolve_sources_returns_stored_values_and_rejects_unknown():
    with tempfile.TemporaryDirectory() as tmp:
        retriever = _retriever(cache_dir=Path(tmp))
        assert retriever.resolve_sources(["甲书.pdf"]) == ["/Users/x/data/甲书.pdf"]  # 不重建路径，用库里的原值
        assert sorted(retriever.resolve_sources(["乙书.pdf", "甲书.pdf", "乙书.pdf"])) == sorted(
            ["/Users/x/data/乙书.pdf", "/other/dir/乙书.pdf", "/Users/x/data/甲书.pdf"]
        )
        try:
            retriever.resolve_sources(["甲书.pdf", "不存在.pdf", "也没有.pdf"])
        except UnknownSourceError as exc:
            assert exc.unknown == ["不存在.pdf", "也没有.pdf"] and exc.valid == ["丙书.md", "乙书.pdf", "甲书.pdf"]
            assert "不存在.pdf" in str(exc)
        else:
            raise AssertionError("未知书名应抛 UnknownSourceError")
        try:
            retriever.invoke("关键路径", sources=["不存在.pdf"])
        except UnknownSourceError:
            pass
        else:
            raise AssertionError("invoke 也应校验书名")


def test_bm25_route_respects_sources_and_k_larger_than_filtered_corpus():
    with tempfile.TemporaryDirectory() as tmp:
        retriever = _retriever(cache_dir=Path(tmp), final_k=20, per_route_k=30)
        unfiltered = retriever.invoke("关键路径", mode="bm25")
        assert set(_ids(unfiltered)) == {"d1", "d3", "d5"}
        only_jia = retriever.invoke("关键路径", mode="bm25", sources=["甲书.pdf"])
        assert _ids(only_jia) == ["d1"]  # k=20 远大于过滤后的 2 个片段，也不报错
        only_yi = retriever.invoke("关键路径", mode="bm25", sources=["乙书.pdf"])
        assert set(_ids(only_yi)) == {"d3", "d5"}  # 同名不同目录的两处都算
        assert all(doc.metadata["bm25_rank"] and doc.metadata["vec_rank"] is None for doc in only_yi)


def test_vector_route_passes_where_filter_and_hybrid_respects_it():
    with tempfile.TemporaryDirectory() as tmp:
        collection = _Collection()
        retriever = _retriever(collection, cache_dir=Path(tmp))
        docs = retriever.invoke("关键路径", mode="vector", sources=["甲书.pdf", "丙书.md"])
        assert collection.query_kwargs[-1] == {"source": {"$in": ["/Users/x/data/甲书.pdf", "/Users/x/data/丙书.md"]}}
        assert set(_ids(docs)) == {"d1", "d2", "d4"}
        hybrid = retriever.invoke("关键路径", mode="hybrid", sources=["乙书.pdf"])
        assert set(_ids(hybrid)) == {"d3", "d5"}
        retriever.invoke("关键路径", mode="vector")
        assert collection.query_kwargs[-1] is None  # 不过滤时不传 where
        retriever.invoke("关键路径", mode="vector", sources=[])
        assert collection.query_kwargs[-1] is None  # 空列表等于不过滤


def test_filter_matching_nothing_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        retriever = _retriever(cache_dir=Path(tmp))
        # 查询词只出现在甲书 / 乙书里，限定丙书 -> BM25 路没有命中
        assert retriever.invoke("关键路径", mode="bm25", sources=["丙书.md"]) == []
        # 纯向量：书在快照里，但库里已没有它的片段 -> where 零匹配 -> 空
        stale = _Collection()
        retriever2 = _retriever(stale, cache_dir=Path(tmp))
        stale.rows = [row for row in stale.rows if row[0] != "d4"]
        assert retriever2.invoke("红烧肉", mode="vector", sources=["丙书.md"]) == []
        assert retriever2._vector_route("q", 5, ["/nonexistent.pdf"]) == []
        # 过滤后语料为 0 的掩码路径直接返回空，不去调 bm25s
        snapshot = retriever._snapshot
        assert retriever._bm25_route(snapshot, "关键路径", 10, []) == []


def test_vector_only_retriever_builds_catalog_lazily_for_sources():
    with tempfile.TemporaryDirectory() as tmp:
        collection = _Collection()
        retriever = _retriever(collection, cache_dir=Path(tmp), mode="vector")
        assert not retriever.bm25_ready and retriever._snapshot is None  # 构造时什么都不读
        docs = retriever.invoke("关键路径", sources=["丙书.md"])
        assert _ids(docs) == ["d4"] and not retriever.bm25_ready
        assert [entry["name"] for entry in retriever.catalog()] == ["丙书.md", "乙书.pdf", "甲书.pdf"]
        assert retriever.bm25_built_at is None and retriever.bm25_chunk_count is None


def test_real_chroma_where_filter_with_k_larger_than_filtered_count():
    with tempfile.TemporaryDirectory() as tmp:
        client = chromadb.PersistentClient(path=str(Path(tmp) / "chroma"))
        collection = client.get_or_create_collection("rag")
        collection.upsert(
            ids=[row[0] for row in LEGACY],
            embeddings=[[float(i), 1.0] for i in range(len(LEGACY))],
            documents=[row[1] for row in LEGACY],
            metadatas=[row[2] for row in LEGACY],
        )
        retriever = _retriever(
            collection, cache_dir=Path(tmp) / "cache", embeddings=_Embeddings((1.0, 1.0)),
            mode="vector", build_bm25=True, final_k=20,
        )
        docs = retriever.invoke("anything", sources=["甲书.pdf"])  # n_results=20，但过滤后只有 2 个
        assert set(_ids(docs)) == {"d1", "d2"}
        assert all(doc.metadata["source"].endswith("甲书.pdf") for doc in docs)
        assert retriever._vector_route("q", 5, ["/no/such/source.pdf"]) == []  # where 零匹配 -> 空
        hybrid = retriever.invoke("关键路径", mode="hybrid", sources=["乙书.pdf"])
        assert set(_ids(hybrid)) == {"d3", "d5"}


def test_refresh_swaps_snapshot_atomically_and_updates_corpus():
    with tempfile.TemporaryDirectory() as tmp:
        collection = _Collection()
        retriever = _retriever(collection, cache_dir=Path(tmp))
        old_snapshot, old_built_at = retriever._snapshot, retriever.bm25_built_at
        assert retriever.bm25_chunk_count == 5 and old_built_at is not None
        assert retriever.invoke("PMBOK", mode="bm25") == []

        collection.rows.append(("d6", "PMBOK 指南的知识领域", {"source": "/Users/x/data/丁书.pdf", "page": 0}))
        assert retriever.invoke("PMBOK", mode="bm25") == []  # 刷新前仍是旧快照
        retriever.refresh()
        assert retriever._snapshot is not old_snapshot and retriever.bm25_chunk_count == 6
        assert _ids(retriever.invoke("PMBOK", mode="bm25")) == ["d6"]
        assert "丁书.pdf" in [entry["name"] for entry in retriever.catalog()]
        assert retriever.bm25_built_at >= old_built_at
        assert old_snapshot.bm25 is not None and len(old_snapshot.ids) == 5  # 旧快照不可变，仍然完整


def test_searches_during_refresh_always_see_a_complete_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        collection = _Collection()
        retriever = _retriever(collection, cache_dir=Path(tmp))
        errors, counts = [], set()
        stop = threading.Event()

        def searcher():
            while not stop.is_set():
                try:
                    docs = retriever.invoke("关键路径", mode="bm25")
                    counts.add(len(docs))
                    assert retriever.bm25_chunk_count in (5, 6, 7)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        threads = [threading.Thread(target=searcher) for _ in range(3)]
        for thread in threads:
            thread.start()
        try:
            for index in range(2):
                collection.rows.append((f"n{index}", f"新增的关键路径内容{index}", {"source": "/d/新书.pdf"}))
                retriever.refresh()
        finally:
            stop.set()
            for thread in threads:
                thread.join()
        assert not errors, errors
        assert counts <= {3, 4, 5}, counts  # 命中数只会是 3（旧）、4、5（新），不会出现半成品
        assert retriever.bm25_chunk_count == 7


def test_heading_path_is_tokenized_into_bm25_but_text_is_returned_plain():
    rows = [
        ("h1", "这一页只有无关的正文内容", {"source": "/d/书.pdf", "heading_path": "第11章 项目成本管理 > 11.3 挣值分析"}),
        ("h2", "另一页讲的是别的东西", {"source": "/d/书.pdf"}),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        retriever = _retriever(_Collection(rows), cache_dir=Path(tmp))
        docs = retriever.invoke("挣值分析", mode="bm25")
        assert _ids(docs) == ["h1"]
        assert docs[0].page_content == "这一页只有无关的正文内容"  # 返回正文，不含标题
        assert docs[0].metadata["heading_path"] == "第11章 项目成本管理 > 11.3 挣值分析"


def test_token_cache_hit_invalidation_and_stale_file_cleanup():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        collection = _Collection()
        first = _retriever(collection, cache_dir=cache_dir)
        files = sorted(cache_dir.glob("tokens-*.json"))
        assert first.cache_hit is False and len(files) == 1
        assert _retriever(collection, cache_dir=cache_dir).cache_hit is True  # 内容没变 -> 命中

        # 同样的 ID 和正文，只是多了章节路径 -> 键变化，旧缓存被清掉
        collection.rows[0] = (*collection.rows[0][:2], {**collection.rows[0][2], "heading_path": "第1章 > 1.1 节"})
        second = _retriever(collection, cache_dir=cache_dir)
        new_files = sorted(cache_dir.glob("tokens-*.json"))
        assert second.cache_hit is False and len(new_files) == 1 and new_files != files

        # 新增片段 -> 再次失效；残留的 .tmp 也被清理
        (cache_dir / "tokens-garbage.tmp").write_text("x", encoding="utf-8")
        collection.rows.append(("d9", "新片段", {"source": "/d/x.pdf"}))
        third = _retriever(collection, cache_dir=cache_dir)
        assert third.cache_hit is False and len(list(cache_dir.glob("tokens-*"))) == 1

        # 缓存文件损坏 -> 重建而不是报错
        only = next(cache_dir.glob("tokens-*.json"))
        only.write_text("{坏掉了", encoding="utf-8")
        assert _retriever(collection, cache_dir=cache_dir).cache_hit is False


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
