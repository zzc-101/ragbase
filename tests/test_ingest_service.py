"""ragbase.ingest.service 的单元测试：临时数据目录 + 临时 Chroma 向量库（ChromaStore）+ 确定性假 embedding（不加载模型、不访问网络、不碰真实数据）。

运行：uv run python tests/test_ingest_service.py
"""
import hashlib
import logging
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import pymupdf

from ragbase.ingest.service import delete_document, plan_changes, run_ingest
from ragbase.ingest.common import COLLECTION, file_id, load_ingest_meta, save_ingest_meta
from ragbase.vectorstores.chroma import ChromaStore


def _vector(text: str) -> list[float]:
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return [byte / 255 for byte in digest[:8]]


class _Embedder:
    """确定性假 embedding，记录每次被调用时的输入；可选检查调用时锁是否被持有。"""

    def __init__(self, lock=None):
        self.batches: list[list[str]] = []
        self.lock = lock
        self.held_when_called: list[bool] = []

    def __call__(self, texts):
        self.batches.append(list(texts))
        if self.lock is not None:
            self.held_when_called.append(self.lock.depth > 0)
        return [_vector(text) for text in texts]

    @property
    def texts(self):
        return [text for batch in self.batches for text in batch]


class _CountingLock:
    """记录加锁次数、最大嵌套深度的假锁。"""

    def __init__(self):
        self.acquired = 0
        self.depth = 0
        self.max_depth = 0

    def __enter__(self):
        self.acquired += 1
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        return self

    def __exit__(self, *exc):
        self.depth -= 1
        return False


def _paragraphs(tag: str, count: int) -> str:
    return "\n\n".join(f"{tag}第{i}段：" + f"{tag}{i}的内容" * 25 for i in range(count))


@contextmanager
def _workspace():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_dir = root / "data"
        data_dir.mkdir()
        store = ChromaStore(root / "chroma", COLLECTION)
        yield data_dir, store, root / "meta.json", store


def _by_name(store) -> dict[str, list[str]]:
    """文件名 -> 该文件的 chunk ID 列表。"""
    result: dict[str, list[str]] = {}
    for record in store.iter_all():
        result.setdefault(Path(record.metadata["source"]).name, []).append(record.id)
    return result


def _texts(store, ids) -> dict[str, str]:
    """指定 ID 的正文。"""
    wanted = set(ids)
    return {record.id: record.text for record in store.iter_all() if record.id in wanted}


def _make_pdf(path: Path, pages: int, toc: list | None):
    document = pymupdf.open()
    for number in range(1, pages + 1):
        page = document.new_page()
        page.insert_text((72, 100), f"第{number}页的项目管理正文", fontname="china-s")
    if toc:
        document.set_toc(toc)
    document.save(str(path))
    document.close()


def test_first_ingest_adds_chunks_and_writes_meta():
    with _workspace() as (data_dir, store, meta_path, _):
        (data_dir / "a.md").write_text(_paragraphs("甲", 6), encoding="utf-8")
        (data_dir / "b.txt").write_text(_paragraphs("乙", 4), encoding="utf-8")
        (data_dir / "README.md").write_text("说明文件不入库", encoding="utf-8")
        embedder, events = _Embedder(), []
        result = run_ingest(data_dir, store, embedder, meta_path, progress=lambda e, **i: events.append((e, i)))
        groups = _by_name(store)
        assert set(groups) == {"a.md", "b.txt"}
        assert result.chunks_added == store.count() == sum(len(ids) for ids in groups.values()) > 2
        assert result.files_total == result.files_done == 2 and result.reason is None and result.changed
        meta = load_ingest_meta(meta_path)
        assert {info["name"] for info in meta.values()} == {"a.md", "b.txt"}
        assert set(meta) == {file_id(Path("a.md")), file_id(Path("b.txt"))}
        records = list(store.iter_all(with_vectors=True))
        assert all(r.metadata["fid"] in meta and "chunk_index" in r.metadata for r in records)
        assert {r.text for r in records} == set(embedder.texts)  # 无标题时向量化输入 == 正文
        assert len(records[0].vector) == 8
        names = [event for event, _ in events]
        assert names[0] == "plan" and names.count("file_start") == 2 and names.count("file_done") == 2
        # 没变化时再跑一次：不重新向量化、不改索引
        again = _Embedder()
        second = run_ingest(data_dir, store, again, meta_path, progress=lambda e, **i: events.append((e, i)))
        assert again.batches == [] and not second.changed and second.unchanged == 2 and events[-1][0] == "nochange"


def test_modified_file_only_reindexes_that_file_and_removes_stale_chunks():
    with _workspace() as (data_dir, store, meta_path, _):
        a, b = data_dir / "a.md", data_dir / "b.md"
        a.write_text(_paragraphs("甲", 8), encoding="utf-8")
        b.write_text(_paragraphs("乙", 5), encoding="utf-8")
        run_ingest(data_dir, store, _Embedder(), meta_path)
        before = _by_name(store)
        b_before = _texts(store, before["b.md"])

        a.write_text(_paragraphs("丙", 2), encoding="utf-8")  # 内容变短、大小变了 -> 指纹变
        embedder = _Embedder()
        result = run_ingest(data_dir, store, embedder, meta_path)
        after = _by_name(store)
        assert result.files_total == 1 and result.unchanged == 1
        assert all("丙" in text for text in embedder.texts) and embedder.texts  # 只向量化了 a.md 的新内容
        assert len(after["a.md"]) < len(before["a.md"]) and result.chunks_deleted == len(before["a.md"]) - len(after["a.md"])
        assert after["b.md"] == before["b.md"]  # b.md 的片段原封未动
        assert _texts(store, before["b.md"]) == b_before
        docs = list(_texts(store, after["a.md"]).values())
        assert all("甲" not in doc for doc in docs)


def test_deleted_file_removes_its_chunks_and_meta_entry():
    with _workspace() as (data_dir, store, meta_path, _):
        (data_dir / "a.md").write_text(_paragraphs("甲", 4), encoding="utf-8")
        (data_dir / "b.md").write_text(_paragraphs("乙", 4), encoding="utf-8")
        run_ingest(data_dir, store, _Embedder(), meta_path)
        total = store.count()
        a_count = len(_by_name(store)["a.md"])
        (data_dir / "a.md").unlink()
        events = []
        embedder = _Embedder()
        result = run_ingest(data_dir, store, embedder, meta_path, progress=lambda e, **i: events.append((e, i)))
        assert embedder.batches == []
        assert set(_by_name(store)) == {"b.md"} and store.count() == total - a_count
        assert result.chunks_deleted == a_count and result.files_total == 1
        assert [info["name"] for info in load_ingest_meta(meta_path).values()] == ["b.md"]
        assert ("file_deleted", {"name": "a.md", "chunks_deleted": a_count}) in events


def test_legacy_index_is_migrated_by_deleting_ids_not_the_whole_store():
    with _workspace() as (data_dir, store, meta_path, _):
        # 旧版索引：UUID 形式的 ID，没有 fid / heading_path，source 是别处的绝对路径
        old_ids = [f"00000000-0000-0000-0000-{i:012d}" for i in range(7)]
        store.upsert(
            ids=old_ids,
            embeddings=[_vector(f"旧{i}") for i in range(7)],
            documents=[f"旧片段{i}" for i in range(7)],
            metadatas=[{"source": "/old/place/gone.pdf", "page": i} for i in range(7)],
        )
        (data_dir / "new.md").write_text(_paragraphs("新", 3), encoding="utf-8")

        def boom(*args, **kwargs):
            raise AssertionError("服务路径不应清空整个向量库")

        store.reset = boom
        events = []
        result = run_ingest(data_dir, store, _Embedder(), meta_path, progress=lambda e, **i: events.append((e, i)))
        assert result.reason == "legacy" and result.changed
        assert ("migrate", {"chunks": 7}) in events
        assert not set(old_ids) & set(store.all_ids())  # 旧片段全部清掉
        assert set(_by_name(store)) == {"new.md"} and result.chunks_deleted == 7
        assert store.count() == result.chunks_added  # 同一个 store 对象依旧可用
        assert [info["name"] for info in load_ingest_meta(meta_path).values()] == ["new.md"]


def test_index_missing_rebuilds_everything_and_drops_stale_meta():
    with _workspace() as (data_dir, store, meta_path, _):
        (data_dir / "a.md").write_text(_paragraphs("甲", 3), encoding="utf-8")
        # 有旧元数据（含已不存在的文件），但索引是空的
        save_ingest_meta({
            file_id(Path("a.md")): {"name": "a.md", "fp": "stale"},
            file_id(Path("gone.md")): {"name": "gone.md", "fp": "x"},
        }, meta_path)
        assert store.count() == 0
        result = run_ingest(data_dir, store, _Embedder(), meta_path)
        assert result.reason == "index_missing" and result.chunks_added == store.count() > 0
        assert [info["name"] for info in load_ingest_meta(meta_path).values()] == ["a.md"]
        assert plan_changes(load_ingest_meta(meta_path), load_ingest_meta(meta_path), store.count())[3] is None


def test_lock_is_taken_per_batch_and_around_embedding():
    with _workspace() as (data_dir, store, meta_path, _):
        (data_dir / "a.md").write_text(_paragraphs("甲", 40), encoding="utf-8")
        lock = _CountingLock()
        embedder = _Embedder(lock)
        result = run_ingest(data_dir, store, embedder, meta_path, lock=lock, batch_size=2)
        batches = -(-result.chunks_added // 2)
        assert batches >= 3, result  # 确实分成了多批
        assert len(embedder.batches) == batches and all(len(batch) <= 2 for batch in embedder.batches)
        assert lock.acquired == batches  # 每批一次（这次没有要删除的陈旧片段）
        assert all(embedder.held_when_called)  # 向量化发生在持锁期间
        assert lock.depth == 0 and lock.max_depth == 1  # 批间释放、不嵌套

        # 修改文件触发删除陈旧片段：删除批次同样单独加锁
        (data_dir / "a.md").write_text(_paragraphs("乙", 2), encoding="utf-8")
        before = lock.acquired
        second = run_ingest(data_dir, store, _Embedder(lock), meta_path, lock=lock, batch_size=2)
        assert second.chunks_deleted > 0 and lock.acquired > before and lock.depth == 0


def test_real_threading_lock_is_released_between_batches():
    with _workspace() as (data_dir, store, meta_path, _):
        (data_dir / "a.md").write_text(_paragraphs("甲", 30), encoding="utf-8")
        lock = threading.Lock()
        states = []

        def embed(texts):
            states.append(lock.locked())  # 调用时应持锁
            return [_vector(text) for text in texts]

        run_ingest(data_dir, store, embed, meta_path, lock=lock, batch_size=2)
        assert states and all(states) and not lock.locked()


def test_pdf_headings_go_into_embedding_input_but_documents_stay_plain():
    with _workspace() as (data_dir, store, meta_path, _):
        _make_pdf(data_dir / "book.pdf", 2, [[1, "第1章 概述", 1], [2, "1.1 范围", 2]])
        _make_pdf(data_dir / "plain.pdf", 1, None)
        embedder = _Embedder()
        run_ingest(data_dir, store, embedder, meta_path)
        by_source = {}
        for record in store.iter_all():
            by_source.setdefault(Path(record.metadata["source"]).name, []).append((record.text, record.metadata))
        book = sorted(by_source["book.pdf"], key=lambda item: item[1]["page"])
        assert book[0][1]["heading_path"] == "第1章 概述" and book[1][1]["heading_path"] == "第1章 概述 > 1.1 范围"
        assert all("第1章" not in document for document, _ in book)  # document 是纯正文
        assert "第1章 概述\n" + book[0][0] in embedder.texts  # 向量化输入 = 标题 + 换行 + 正文
        assert "第1章 概述 > 1.1 范围\n" + book[1][0] in embedder.texts
        assert all("heading_path" not in metadata for _, metadata in by_source["plain.pdf"])
        assert by_source["plain.pdf"][0][0] in embedder.texts


def test_failed_file_is_reported_and_retried_without_blocking_others():
    with _workspace() as (data_dir, store, meta_path, _):
        (data_dir / "good.md").write_text(_paragraphs("甲", 3), encoding="utf-8")
        (data_dir / "bad.docx").write_bytes(b"this is not a docx")
        (data_dir / "old.doc").write_bytes(b"legacy")
        events = []
        logging.getLogger("ragbase.ingest.service").disabled = True  # 预期内的失败，别在测试输出里刷堆栈
        try:
            result = run_ingest(data_dir, store, _Embedder(), meta_path, progress=lambda e, **i: events.append((e, i)))
        finally:
            logging.getLogger("ragbase.ingest.service").disabled = False
        assert list(result.failed_files) == ["bad.docx"] and result.skipped_files == ["old.doc"]
        assert set(_by_name(store)) == {"good.md"}
        assert [info["name"] for info in load_ingest_meta(meta_path).values()] == ["good.md"]  # 失败文件不记入元数据
        assert result.files_done == result.files_total == 2
        assert any(event == "file_failed" for event, _ in events)
        (data_dir / "bad.docx").unlink()
        (data_dir / "bad.md").write_text(_paragraphs("丁", 2), encoding="utf-8")
        retry = run_ingest(data_dir, store, _Embedder(), meta_path)
        assert not retry.failed_files and set(_by_name(store)) == {"good.md", "bad.md"}


def test_no_documents_does_not_touch_a_legacy_index():
    with _workspace() as (data_dir, store, meta_path, _):
        store.upsert(ids=["x"], embeddings=[_vector("x")], documents=["旧"], metadatas=[{"source": "/o/a.pdf"}])
        events = []
        result = run_ingest(data_dir, store, _Embedder(), meta_path, progress=lambda e, **i: events.append(e))
        assert events == ["no_documents"] and store.count() == 1 and not result.changed
        assert not meta_path.exists()


def test_delete_document_by_fid_and_by_legacy_source():
    with _workspace() as (data_dir, store, meta_path, _):
        (data_dir / "a.md").write_text(_paragraphs("甲", 3), encoding="utf-8")
        (data_dir / "b.md").write_text(_paragraphs("乙", 3), encoding="utf-8")
        run_ingest(data_dir, store, _Embedder(), meta_path)
        b_ids = _by_name(store)["b.md"]
        count = delete_document("a.md", data_dir, store, meta_path)
        assert count > 0 and not (data_dir / "a.md").exists()
        assert set(_by_name(store)) == {"b.md"} and _by_name(store)["b.md"] == b_ids
        assert [info["name"] for info in load_ingest_meta(meta_path).values()] == ["b.md"]

        # 旧版片段（无 fid）按 source 原始值删除
        store.upsert(
            ids=["legacy-1", "legacy-2"],
            embeddings=[_vector("1"), _vector("2")],
            documents=["旧1", "旧2"],
            metadatas=[{"source": "/elsewhere/old.pdf"}, {"source": "/elsewhere/old.pdf"}],
        )
        assert delete_document("old.pdf", data_dir, store, meta_path, sources=["/elsewhere/old.pdf"]) == 2
        assert set(_by_name(store)) == {"b.md"}


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
