"""ragbase.vectorstores 的测试：同一套契约测试同时跑 ChromaStore 和 QdrantStore（本地模式），再测迁移与工厂。

全部在临时目录里用很小的假向量运行，不加载模型、不访问网络、不碰真实数据。

运行：uv run python tests/test_vectorstores.py
"""
import random
import re
import tempfile
import threading
from pathlib import Path

from ragbase.config import Settings
from ragbase.vectorstores import clean_metadata, create_vector_store, store_count
from ragbase.vectorstores.base import VectorStore
from ragbase.vectorstores.chroma import ChromaStore
from ragbase.vectorstores.migrate import MigrationError, migrate_store
from ragbase.vectorstores.qdrant import QdrantStore, point_id

SRC = Path(__file__).resolve().parents[1] / "src" / "ragbase"

BACKENDS = {
    "chroma": lambda root: ChromaStore(root / "chroma_db", "rag"),
    "qdrant": lambda root: QdrantStore("rag", path=root / "qdrant_db"),
}

# 二维向量，查询 [1, 0] 时的相似度顺序固定：a > b > c > d
ROWS = [
    ("a_0", [1.0, 0.0], "甲书第一段", {"fid": "a", "source": "/data/甲书.pdf", "page": 0, "chunk_index": 0}),
    ("a_1", [0.9, 0.436], "甲书第二段", {"fid": "a", "source": "/data/甲书.pdf", "page": 1, "chunk_index": 1}),
    ("b_0", [0.6, 0.8], "乙书第一段", {"fid": "b", "source": "/other/乙书.pdf", "page": 0, "chunk_index": 0}),
    ("legacy-1", [0.0, 1.0], "旧版片段", {"source": "/legacy/dir/old.pdf", "page": 3}),
]


def _fill(store: VectorStore, rows=ROWS) -> None:
    store.upsert([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows])


def _each_backend(check) -> None:
    """对每个后端在独立临时目录里跑 check(store)。"""
    for name, make in BACKENDS.items():
        with tempfile.TemporaryDirectory() as tmp:
            store = make(Path(tmp))
            try:
                check(store)
            except AssertionError as exc:
                raise AssertionError(f"[{name}] {exc}") from exc
            finally:
                store.close()


# ---------------------------------------------------------------- 契约测试（两个后端都要通过）

def test_empty_store_reads_as_empty_and_creates_nothing():
    def check(store):
        assert store.count() == 0 and not store.exists()
        assert store.query([1.0, 0.0], 5) == [] and list(store.iter_all()) == []
        assert store.ids_where(fid="a") == [] and store.all_ids() == []
        store.delete(["nope"])  # 不存在也不报错
        store.reset()
        assert not Path(store.location).exists()  # 只读探测不会建目录

    _each_backend(check)


def test_upsert_count_exists_and_query_order():
    def check(store):
        _fill(store)
        assert store.exists() and store.count() == 4
        hits = store.query([1.0, 0.0], 10)
        assert [h.id for h in hits] == ["a_0", "a_1", "b_0", "legacy-1"]
        assert [h.text for h in hits][:2] == ["甲书第一段", "甲书第二段"]
        scores = [h.score for h in hits]
        # Chroma 是距离（升序），Qdrant 是余弦相似度（降序）；调用方只用顺序
        assert scores == sorted(scores) or scores == sorted(scores, reverse=True), scores
        assert [h.id for h in store.query([1.0, 0.0], 2)] == ["a_0", "a_1"]  # 只取前 k 个
        assert [h.id for h in store.query([0.0, 1.0], 1)] == ["legacy-1"]
        assert store.query([1.0, 0.0], 0) == []

    _each_backend(check)


def test_query_sources_filter():
    def check(store):
        _fill(store)
        only_jia = store.query([1.0, 0.0], 10, sources=["/data/甲书.pdf"])  # k 远大于过滤后的数量
        assert [h.id for h in only_jia] == ["a_0", "a_1"]
        both = store.query([0.0, 1.0], 10, sources=["/other/乙书.pdf", "/legacy/dir/old.pdf"])
        assert [h.id for h in both] == ["legacy-1", "b_0"]
        assert store.query([1.0, 0.0], 10, sources=["/no/such.pdf"]) == []  # 零匹配 -> 空
        assert store.query([1.0, 0.0], 10, sources=[]) == []  # 空列表 = 什么都不匹配
        assert len(store.query([1.0, 0.0], 10, sources=None)) == 4  # None 才是不过滤

    _each_backend(check)


def test_ids_where_fid_sources_and_union():
    def check(store):
        _fill(store)
        assert sorted(store.ids_where(fid="a")) == ["a_0", "a_1"]
        assert store.ids_where(fid="zzz") == []
        assert store.ids_where(sources=["/legacy/dir/old.pdf"]) == ["legacy-1"]
        assert sorted(store.ids_where(fid="b", sources=["/legacy/dir/old.pdf"])) == ["b_0", "legacy-1"]  # 满足任一条件
        assert sorted(store.ids_where(fid="a", sources=["/data/甲书.pdf"])) == ["a_0", "a_1"]  # 去重
        assert store.ids_where() == [] and store.ids_where(sources=[]) == []  # 没有条件 -> 空
        assert sorted(store.all_ids()) == sorted(r[0] for r in ROWS)

    _each_backend(check)


def test_delete_ids():
    def check(store):
        _fill(store)
        store.delete(["a_0", "does-not-exist"])
        assert store.count() == 3 and sorted(store.all_ids()) == ["a_1", "b_0", "legacy-1"]
        store.delete([])
        store.delete(["a_1", "b_0", "legacy-1"])
        assert store.count() == 0

    _each_backend(check)


def test_delete_many_ids_in_batches():
    def check(store):
        ids = [f"x_{i}" for i in range(2300)]
        store.upsert(ids, [[1.0, float(i % 7)] for i in range(2300)], ["t"] * 2300, [{"fid": "x"}] * 2300)
        assert store.count() == 2300 and len(store.ids_where(fid="x")) == 2300  # ids_where 自己翻页
        store.delete(ids[:2250])
        assert store.count() == 50

    _each_backend(check)


def test_upsert_overwrites_same_id():
    def check(store):
        _fill(store)
        store.upsert(["a_0"], [[0.0, 1.0]], ["改过的正文"], [{"fid": "a", "source": "/data/新路径.pdf", "chunk_index": 0}])
        assert store.count() == 4  # 同 ID 覆盖，不新增
        record = next(r for r in store.iter_all() if r.id == "a_0")
        assert record.text == "改过的正文" and record.metadata["source"] == "/data/新路径.pdf"
        # 注意：Chroma 的 upsert 按键合并元数据（旧键保留），Qdrant 整体替换；入库每次都写完整元数据，所以不受影响
        assert store.query([0.0, 1.0], 1)[0].id in ("a_0", "legacy-1")  # 向量也变了
        assert [h.id for h in store.query([1.0, 0.0], 1)] == ["a_1"]

    _each_backend(check)


def test_metadata_and_text_round_trip_including_non_ascii():
    def check(store):
        metadata = {
            "source": "/数据/项目管理 知识体系(第7版).pdf", "heading_path": "第11章 风险管理 > 11.3 “定性”分析",
            "page": 12, "ratio": 0.25, "flag": True, "fid": "f1", "content_hash": "abc123",
        }
        text = "挣值分析：CPI < 1 表示成本超支。\n第二行 \"引号\" 和 emoji-free 文本。"
        store.upsert(["c_1"], [[1.0, 0.0]], [text], [metadata])
        record = next(iter(store.iter_all()))
        assert record.text == text and record.metadata == metadata, record
        assert type(record.metadata["page"]) is int and record.metadata["flag"] is True
        hit = store.query([1.0, 0.0], 1)[0]
        assert hit.id == "c_1" and hit.text == text and hit.metadata == metadata
        assert store.ids_where(sources=[metadata["source"]]) == ["c_1"]  # 非 ASCII 路径也能过滤

    _each_backend(check)


def test_iter_all_pagination_and_vectors():
    def check(store):
        rng = random.Random(7)
        vectors = {f"id-{i:03d}": [rng.random() for _ in range(4)] for i in range(25)}
        ids = list(vectors)
        store.upsert(ids, [vectors[i] for i in ids], [f"文本{i}" for i in ids], [{"n": n} for n in range(25)])
        seen = list(store.iter_all(batch_size=7))
        assert sorted(r.id for r in seen) == ids and len({r.id for r in seen}) == 25  # 不重不漏
        assert all(r.vector is None for r in seen)
        assert {r.id: r.text for r in seen}[ids[3]] == "文本id-003"
        with_vectors = {r.id: r.vector for r in store.iter_all(batch_size=10, with_vectors=True)}
        assert len(with_vectors) == 25
        for doc_id, vector in with_vectors.items():
            assert len(vector) == 4
            # 向量保真（Qdrant 余弦会归一化，所以比较方向）
            dot = sum(a * b for a, b in zip(vector, vectors[doc_id]))
            norm = (sum(a * a for a in vector) * sum(b * b for b in vectors[doc_id])) ** 0.5
            assert abs(dot / norm - 1) < 1e-5, doc_id
        assert len(list(store.iter_all(batch_size=25))) == 25 and len(list(store.iter_all(batch_size=1000))) == 25

    _each_backend(check)


def test_reset_drops_everything_and_store_is_reusable():
    def check(store):
        _fill(store)
        store.reset()
        assert store.count() == 0 and not store.exists()
        assert store.query([1.0, 0.0], 3) == [] and list(store.iter_all()) == []
        _fill(store, ROWS[:2])  # 重置后同一个对象还能继续写
        assert store.count() == 2 and store.exists()
        store.reset()
        store.reset()  # 重复重置无副作用
        assert store.count() == 0

    _each_backend(check)


def test_data_persists_after_close_and_reopen():
    for name, make in BACKENDS.items():
        with tempfile.TemporaryDirectory() as tmp:
            first = make(Path(tmp))
            _fill(first)
            first.close()
            second = make(Path(tmp))
            assert second.count() == 4, name
            assert [h.id for h in second.query([1.0, 0.0], 2)] == ["a_0", "a_1"], name
            second.close()
            first.close()  # 重复关闭无副作用


def test_concurrent_reads_and_writes():
    def check(store):
        _fill(store)
        errors = []

        def writer(tag):
            try:
                for i in range(15):
                    store.upsert([f"{tag}_{i}"], [[1.0, float(i)]], ["文本"], [{"fid": tag}])
                    store.delete([f"{tag}_{i - 1}"]) if i % 3 == 0 else None
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def reader():
            try:
                for _ in range(20):
                    store.query([1.0, 0.0], 3)
                    store.count()
                    store.ids_where(fid="a")
                    list(store.iter_all(batch_size=5))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(tag,)) for tag in ("w1", "w2")] + [
            threading.Thread(target=reader) for _ in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors, errors
        assert len(store.ids_where(fid="w1")) == store.count() - 4 - len(store.ids_where(fid="w2"))

    _each_backend(check)


# ---------------------------------------------------------------- Qdrant 专有行为

def test_qdrant_id_mapping_is_stable_and_round_trips():
    ids = ["a_0", "00000000-0000-0000-0000-000000000001", "文件_12", "a b/c:d", "x" * 300]
    points = [point_id(i) for i in ids]
    assert len(set(points)) == len(ids) and points == [point_id(i) for i in ids]  # 稳定且互不相同
    import uuid

    assert all(str(uuid.UUID(p)) == p for p in points)  # 都是合法 UUID
    with tempfile.TemporaryDirectory() as tmp:
        store = QdrantStore("rag", path=Path(tmp) / "q")
        store.upsert(ids, [[1.0, float(i)] for i in range(len(ids))], ["t"] * len(ids), [{"fid": "f"}] * len(ids))
        assert sorted(store.all_ids()) == sorted(ids)  # 原始字符串 ID 原样还原
        assert sorted(h.id for h in store.query([1.0, 0.0], 10)) == sorted(ids)
        store.delete(["文件_12"])
        assert "文件_12" not in store.all_ids() and store.count() == len(ids) - 1
        store.close()


def test_qdrant_payload_layout():
    with tempfile.TemporaryDirectory() as tmp:
        store = QdrantStore("rag", path=Path(tmp) / "q")
        store.upsert(["k_1"], [[1.0, 0.0]], ["正文"], [{"fid": "k", "source": "/s.pdf", "chunk_id": "元数据里的同名键"}])
        client, _ = __import__("ragbase.vectorstores.qdrant", fromlist=["_acquire"])._acquire(store.path, None, None)
        point = client.retrieve("rag", [point_id("k_1")], with_payload=True)[0]
        assert point.payload == {
            "chunk_id": "k_1", "document": "正文",
            "metadata": {"fid": "k", "source": "/s.pdf", "chunk_id": "元数据里的同名键"},  # 元数据嵌套，不会和顶层键冲突
        }
        assert next(iter(store.iter_all())).metadata["chunk_id"] == "元数据里的同名键"
        store.close()


def test_qdrant_local_client_is_shared_per_path_within_a_process():
    with tempfile.TemporaryDirectory() as tmp:
        first = QdrantStore("rag", path=Path(tmp) / "q")
        second = QdrantStore("rag", path=Path(tmp) / "q")  # 同目录再开一个：共享同一个 client，不会报“目录已被占用”
        _fill(first)
        assert second.count() == 4
        second.upsert(["z_0"], [[0.5, 0.5]], ["z"], [{"fid": "z"}])
        assert first.count() == 5
        other = QdrantStore("rag", path=Path(tmp) / "q2")  # 不同目录互不影响
        assert other.count() == 0 and not other.exists()
        first.close()
        assert second.count() == 5  # 关闭后再用会自动重新打开，数据还在
        second.close()
        other.close()


def test_qdrant_collections_are_isolated_by_name():
    with tempfile.TemporaryDirectory() as tmp:
        one, two = QdrantStore("kb1", path=Path(tmp) / "q"), QdrantStore("kb2", path=Path(tmp) / "q")
        _fill(one)
        assert one.count() == 4 and two.count() == 0 and not two.exists()
        two.upsert(["t"], [[1.0, 0.0]], ["t"], [{}])
        one.reset()
        assert one.count() == 0 and two.count() == 1
        one.close()


def test_qdrant_requires_path_or_url():
    try:
        QdrantStore("rag")
    except ValueError as exc:
        assert "path" in str(exc)
    else:
        raise AssertionError("既没有 path 也没有 url 应报错")


# ---------------------------------------------------------------- 迁移

def _random_unit_rows(count: int, dim: int = 16, seed: int = 3):
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        vector = [rng.gauss(0, 1) for _ in range(dim)]
        norm = sum(x * x for x in vector) ** 0.5
        rows.append((
            f"{'ab'[index % 2]}f_{index}" if index % 5 else f"00000000-0000-0000-0000-{index:012d}",
            [x / norm for x in vector],
            f"第{index}段正文",
            {"fid": f"f{index % 3}", "source": f"/books/书{index % 3}.pdf", "page": index, "chunk_index": index},
        ))
    return rows


def test_migrate_chroma_to_qdrant_preserves_everything_and_ranking():
    rows = _random_unit_rows(60)
    with tempfile.TemporaryDirectory() as tmp:
        source, target = BACKENDS["chroma"](Path(tmp)), BACKENDS["qdrant"](Path(tmp))
        _fill(source, rows)
        progress = []
        result = migrate_store(source, target, batch_size=25, progress=lambda done, total: progress.append((done, total)))
        assert (result.source_count, result.copied, result.target_count) == (60, 60, 60)
        assert progress == [(25, 60), (50, 60), (60, 60)]
        # ID、正文、元数据完全一致
        src = {r.id: (r.text, r.metadata) for r in source.iter_all()}
        dst = {r.id: (r.text, r.metadata) for r in target.iter_all()}
        assert src == dst
        # 同一个查询向量，两个库的前 10 名完全一致（含按书过滤）
        rng = random.Random(11)
        for _ in range(8):
            query = [rng.gauss(0, 1) for _ in range(16)]
            assert [h.id for h in source.query(query, 10)] == [h.id for h in target.query(query, 10)]
            sources = ["/books/书1.pdf", "/books/书2.pdf"]
            assert [h.id for h in source.query(query, 7, sources)] == [h.id for h in target.query(query, 7, sources)]
        assert sorted(source.ids_where(fid="f1")) == sorted(target.ids_where(fid="f1"))
        source.close()
        target.close()


def test_migrate_qdrant_back_to_chroma():
    rows = _random_unit_rows(30, seed=5)
    with tempfile.TemporaryDirectory() as tmp:
        source, target = BACKENDS["qdrant"](Path(tmp)), BACKENDS["chroma"](Path(tmp))
        _fill(source, rows)
        assert migrate_store(source, target).target_count == 30
        query = rows[4][1]
        assert [h.id for h in source.query(query, 5)] == [h.id for h in target.query(query, 5)]
        source.close()
        target.close()


def test_migrate_refuses_non_empty_target_unless_forced_and_other_bad_cases():
    rows = _random_unit_rows(10)
    with tempfile.TemporaryDirectory() as tmp:
        source, target = BACKENDS["chroma"](Path(tmp)), BACKENDS["qdrant"](Path(tmp))
        try:
            migrate_store(source, target)  # 源库不存在
        except MigrationError as exc:
            assert "为空或不存在" in str(exc)
        else:
            raise AssertionError("空源库应拒绝")
        _fill(source, rows)
        _fill(target, [("stale", [1.0] * 16, "旧数据", {"fid": "old"})])
        try:
            migrate_store(source, target)
        except MigrationError as exc:
            assert "--force" in str(exc) and "1 个片段" in str(exc)
        else:
            raise AssertionError("目标非空应拒绝")
        assert target.count() == 1  # 被拒绝时目标库原封不动
        result = migrate_store(source, target, force=True)
        assert result.target_count == 10 and "stale" not in target.all_ids()  # force：先清空目标再复制
        try:
            migrate_store(source, source)
        except MigrationError as exc:
            assert "同一个" in str(exc)
        else:
            raise AssertionError("源与目标相同应拒绝")
        source.close()
        target.close()


# ---------------------------------------------------------------- 工厂与配置

def _settings(root: Path, **kwargs):
    return Settings(_env_file=None, db_dir=root / "chroma_db", qdrant_path=root / "qdrant_db", **kwargs)


def test_factory_default_is_chroma_and_follows_settings():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = create_vector_store(_settings(root, vector_store="chroma"))
        assert isinstance(store, ChromaStore) and store.name == "chroma"
        assert store.location == str(root / "chroma_db") and store.collection_name == "rag"
        named = create_vector_store(_settings(root, vector_store="chroma", collection="kb"))
        assert named.collection_name == "kb"
        qdrant = create_vector_store(_settings(root, vector_store="qdrant", collection="kb"))
        assert isinstance(qdrant, QdrantStore) and qdrant.name == "qdrant"
        assert qdrant.location == str(root / "qdrant_db") and qdrant.collection_name == "kb" and qdrant.url is None
        # 指定 backend 覆盖配置（迁移命令用）
        assert isinstance(create_vector_store(_settings(root, vector_store="chroma"), backend="qdrant"), QdrantStore)
        assert not (root / "chroma_db").exists() and not (root / "qdrant_db").exists()  # 创建对象不建目录


def test_factory_qdrant_server_mode_uses_url_and_secret_key():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp), vector_store="qdrant", qdrant_url="http://qdrant.example:6333", qdrant_api_key="k-123")
        store = create_vector_store(settings)
        assert store.url == "http://qdrant.example:6333" and store.path is None and store.location == store.url
        assert store._api_key == "k-123" and "k-123" not in repr(settings)


def test_factory_unknown_backend_gives_clear_error():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            create_vector_store(_settings(Path(tmp)), backend="milvus")
        except ValueError as exc:
            assert "milvus" in str(exc) and "chroma" in str(exc) and "qdrant" in str(exc) and "RAG_VECTOR_STORE" in str(exc)
        else:
            raise AssertionError("未知后端应报错")
        try:
            _settings(Path(tmp), vector_store="milvus")  # 配置层面就拒绝，服务启动即失败
        except ValueError:
            pass
        else:
            raise AssertionError("配置里的未知后端应被拒绝")


def test_store_count_reports_none_for_missing_index():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for backend in ("chroma", "qdrant"):
            settings = _settings(root, vector_store=backend)
            assert store_count(settings) is None  # 目录不存在
            store = create_vector_store(settings)
            assert store_count(settings) is None  # 对象存在但集合还没创建
            _fill(store)
            assert store_count(settings) == 4
            store.reset()
            assert store_count(settings) is None
            store.close()


# ---------------------------------------------------------------- 命令行（ingest --full / migrate-store）

class _FakeEmbeddings:
    """替身 Qwen3Embeddings：确定性 8 维向量，不加载模型。"""

    def __init__(self, *args, **kwargs):
        pass

    def embed_documents(self, texts):
        import hashlib

        return [[byte / 255 for byte in hashlib.md5(text.encode("utf-8")).digest()[:8]] for text in texts]


def _run_cli(root: Path, argv: list[str], **env) -> tuple[int, str]:
    """在隔离的环境变量里运行 ragbase.cli.main，返回 (退出码, 标准输出)。RAG_HOME 指向临时目录，不会读到真实的 .env。"""
    import contextlib
    import io
    import os

    from ragbase import cli
    from ragbase.config import get_settings

    values = {"RAG_HOME": str(root), "RAG_DB_DIR": str(root / "chroma_db"), "RAG_QDRANT_PATH": str(root / "qdrant_db"),
              "RAG_DATA_DIR": str(root / "data"), "RAG_META_FILE": str(root / "meta.json"), **env}
    saved = {key: value for key, value in os.environ.items() if key.startswith("RAG_")}
    for key in saved:
        del os.environ[key]
    os.environ.update(values)
    get_settings.cache_clear()
    out, code = io.StringIO(), 0
    try:
        with contextlib.redirect_stdout(out):
            try:
                code = cli.main(argv)
            except SystemExit as exc:
                code = exc.code
    finally:
        for key in values:
            os.environ.pop(key, None)
        os.environ.update(saved)
        get_settings.cache_clear()
    return code, out.getvalue()


def test_cli_ingest_full_resets_the_store_instead_of_deleting_directories():
    from ragbase.models import embeddings as embeddings_module

    original = embeddings_module.Qwen3Embeddings
    embeddings_module.Qwen3Embeddings = _FakeEmbeddings
    try:
        for backend in ("chroma", "qdrant"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "data").mkdir()
                for name in "abc":  # 每个文件至少一个片段
                    (root / "data" / f"{name}.md").write_text(f"{name}书第一段：" + "内容" * 40, encoding="utf-8")
                env = {"RAG_VECTOR_STORE": backend}
                code, out = _run_cli(root, ["ingest", "--full"], **env)
                assert code == 0 and "完成" in out, (backend, out)
                settings = _settings(root, vector_store=backend)
                first = store_count(settings)
                assert first and first > 1
                # 在旧索引里塞一条残留数据 + 再来一次 --full：残留数据必须被清掉，其他文件（目录里的无关文件）不受影响
                stale = create_vector_store(settings)
                stale.upsert(["stale"], [[0.0] * 8], ["残留"], [{"fid": "gone"}])
                stale.close()
                (Path(stale.location) / "keep.txt").write_text("别的文件", encoding="utf-8")
                code, out = _run_cli(root, ["ingest", "--full"], **env)
                assert code == 0, (backend, out)
                after = create_vector_store(settings)
                assert "stale" not in after.all_ids() and after.count() == first, backend
                assert (Path(stale.location) / "keep.txt").exists(), backend  # 目录没有被整个删除
                after.close()
                code, out = _run_cli(root, ["ingest"], **env)  # 增量：没有变化
                assert code == 0 and "没有变化" in out, (backend, out)
    finally:
        embeddings_module.Qwen3Embeddings = original


def test_cli_migrate_store_chroma_to_qdrant():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = create_vector_store(_settings(root), backend="chroma")
        _fill(source, _random_unit_rows(12, dim=8))
        source.close()
        code, out = _run_cli(root, ["migrate-store", "--to", "qdrant"])
        assert code == 0 and "目标库现有 12" in out and "RAG_VECTOR_STORE=qdrant" in out, out
        target = create_vector_store(_settings(root), backend="qdrant")
        assert target.count() == 12
        target.close()
        code, out = _run_cli(root, ["migrate-store", "--to", "qdrant"])  # 目标非空 -> 拒绝
        assert code == 1 and "--force" in out, out
        code, out = _run_cli(root, ["migrate-store", "--to", "qdrant", "--from", "qdrant"])
        assert code == 2 and "相同" in out, out
        code, out = _run_cli(root, ["migrate-store", "--to", "qdrant", "--force", "--batch-size", "5"])
        assert code == 0 and "已复制 12/12" in out, out


# ---------------------------------------------------------------- 其他

def test_clean_metadata_keeps_scalars_and_stringifies_the_rest():
    cleaned = clean_metadata({"a": 1, "b": "x", "c": 1.5, "d": True, "e": None, "f": [1, 2], "g": {"k": 1}})
    assert cleaned == {"a": 1, "b": "x", "c": 1.5, "d": True, "f": "[1, 2]", "g": "{'k': 1}"}


def test_backend_libraries_are_imported_only_inside_their_adapter():
    """chromadb 只在 vectorstores/chroma.py 里 import，qdrant_client 只在 vectorstores/qdrant.py 里。"""
    pattern = {
        "chromadb": re.compile(r"^\s*(?:import|from)\s+chromadb\b", re.MULTILINE),
        "qdrant_client": re.compile(r"^\s*(?:import|from)\s+qdrant_client\b", re.MULTILINE),
    }
    allowed = {"chromadb": "vectorstores/chroma.py", "qdrant_client": "vectorstores/qdrant.py"}
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for library, regex in pattern.items():
            if regex.search(text):
                assert path.relative_to(SRC).as_posix() == allowed[library], f"{path} 不应直接 import {library}"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
