"""增量入库服务：命令行（ingest.py / ingest_incremental.py）与 HTTP 服务（POST /ingest）共用。

所有依赖都显式传入（文档目录、Chroma collection、向量化函数、元数据文件、锁、进度回调），
所以测试可以用临时目录 + 假 embedding 运行，不碰真实数据。

写入方式：直接用 collection.upsert / collection.delete（预先算好的向量），不经过 langchain 的 Chroma 封装，
也不删除或重建 collection——运行中的服务持有 collection 对象，删集合会让它失效。

并发（服务内）：传入 pipeline.lock，每个批次（向量化 + 写入，或一批删除）单独加锁、批间释放，
所以入库期间 /search 可以插进来；本地模型不能并发推理，这把锁同时保证了入库与检索不会同时用模型。

元数据（.ingest_meta.json）保存语义：每个文件成功入库（或其旧片段删除完）后立即落盘，
所以中途失败 / 中断时已完成的文件不会重做；失败的文件不写入元数据，下次会重试。

进度回调 progress(event, **info) 的事件：
  no_documents data_dir, skipped       数据目录里没有文档（也没有旧元数据）
  plan   reason, to_add, deleted, unchanged, skipped   增量计划（to_add / deleted 是文件名列表）
  nochange                             没有变化
  migrate chunks                       迁移旧版索引：已清空的旧片段数
  file_start name                      开始处理某个文件
  file_done name, chunks_added, chunks_deleted, report   文件处理完成
  file_deleted name, chunks_deleted    源文件已不存在，旧片段已删除
  file_failed name, error              文件入库失败（其余文件继续）
"""
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from document_parser import list_documents
from rag_common import (
    build_ingest_meta,
    embedding_input,
    file_id,
    load_ingest_meta,
    prepare_file_chunks,
    save_ingest_meta,
)

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 64  # 每批向量化 + 写入的片段数；也是持锁的粒度
DELETE_BATCH_SIZE = 1000  # 每次 collection.delete 的 ID 数（低于 SQLite 变量上限）


@dataclass
class IngestResult:
    """一次入库的结果。changed 表示索引内容是否发生过变化（需要刷新 BM25）。"""

    reason: str | None = None  # None=正常增量 / "index_missing" / "legacy"
    files_total: int = 0
    files_done: int = 0
    unchanged: int = 0
    chunks_added: int = 0
    chunks_deleted: int = 0
    skipped_files: list[str] = field(default_factory=list)  # 被跳过的旧版 .doc
    failed_files: dict[str, str] = field(default_factory=dict)  # 文件名 -> 失败原因
    total_chunks: int | None = None  # 结束时索引里的片段总数
    changed: bool = False


def plan_changes(old_meta: dict, cur_meta: dict, collection_count: int | None):
    """对比旧元数据和当前文件，返回 (to_add, deleted, unchanged, reason)。

    collection_count 为 None 表示索引集合不存在。reason：
    None=正常增量；"index_missing"=有元数据但索引缺失/为空，全量重建；
    "legacy"=旧版索引（有数据但无元数据），迁移为可增量索引。
    """
    index_empty = not collection_count
    if old_meta and index_empty:
        return list(cur_meta), [], 0, "index_missing"
    if not old_meta and not index_empty:
        return list(cur_meta), [], 0, "legacy"

    to_add = []      # fid 列表
    unchanged = 0
    for fid, info in cur_meta.items():
        old = old_meta.get(fid)
        if old is None or old["fp"] != info["fp"]:
            to_add.append(fid)  # 新文件 或 内容变了
        else:
            unchanged += 1
    deleted = [fid for fid in old_meta if fid not in cur_meta]
    return to_add, deleted, unchanged, None


def clean_metadata(metadata: dict) -> dict:
    """Chroma 元数据只接受 str / int / float / bool：去掉 None，其他类型转成字符串。"""
    return {
        key: value if isinstance(value, (str, int, float, bool)) else str(value)
        for key, value in metadata.items()
        if value is not None
    }


def delete_chunk_ids(collection, ids: list[str], lock=None) -> int:
    """按 ID 分批删除片段，每批单独加锁。返回删除数。"""
    lock = nullcontext() if lock is None else lock
    for start in range(0, len(ids), DELETE_BATCH_SIZE):
        with lock:
            collection.delete(ids=ids[start:start + DELETE_BATCH_SIZE])
    return len(ids)


def _chunk_ids_where(collection, where: dict) -> list[str]:
    return collection.get(where=where, include=[])["ids"]


def delete_file_chunks(collection, fid: str | None = None, sources: list[str] | None = None, lock=None) -> int:
    """删除某个文件的全部片段：按 fid 匹配，并（旧版索引没有 fid）按 source 元数据的原始值匹配。"""
    ids: dict[str, None] = {}
    if fid:
        ids.update(dict.fromkeys(_chunk_ids_where(collection, {"fid": fid})))
    if sources:
        ids.update(dict.fromkeys(_chunk_ids_where(collection, {"source": {"$in": list(sources)}})))
    return delete_chunk_ids(collection, list(ids), lock)


def _index_file(path: Path, fid: str, collection, embed_fn, lock, batch_size: int) -> tuple[int, int, dict]:
    """解析并写入一个文件，返回 (新增片段数, 删除的陈旧片段数, 预处理报告)。

    先 upsert 新片段（ID 稳定，同 ID 覆盖），最后才删除不再存在的旧片段，
    所以重建某个文件期间它的内容始终可被检索。
    """
    texts, metadatas, ids, report = prepare_file_chunks(path)
    old_ids = _chunk_ids_where(collection, {"fid": fid})
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batch_inputs = [
            embedding_input(text, metadata)
            for text, metadata in zip(texts[start:end], metadatas[start:end])
        ]
        with lock:
            vectors = embed_fn(batch_inputs)
            collection.upsert(
                ids=ids[start:end],
                embeddings=vectors,
                documents=texts[start:end],  # 存纯正文；章节路径在 metadata.heading_path 里
                metadatas=[clean_metadata(metadata) for metadata in metadatas[start:end]],
            )
    new_ids = set(ids)
    stale = [chunk_id for chunk_id in old_ids if chunk_id not in new_ids]
    delete_chunk_ids(collection, stale, lock)
    return len(texts), len(stale), report


def run_ingest(
    data_dir: Path,
    collection,
    embed_fn,
    meta_path: Path,
    lock=None,
    progress=None,
    batch_size: int = EMBED_BATCH_SIZE,
) -> IngestResult:
    """增量入库：新增 / 修改的文件重新索引，已删除的文件移除片段。

    embed_fn(texts: list[str]) -> list[list[float]]：文档向量化（输入已含章节路径前缀）。
    lock：任何带 __enter__/__exit__ 的锁，None 表示不加锁。
    首次运行（库为空）等价于全量；旧版索引（有片段但没有元数据）会先按 ID 分批清空再重建。
    """
    notify = progress or (lambda event, **info: None)
    lock = nullcontext() if lock is None else lock
    result = IngestResult()

    docs_paths, skipped = list_documents(Path(data_dir))  # 目录不存在时抛 FileNotFoundError，避免误清空索引
    result.skipped_files = [path.name for path in skipped]
    old_meta = load_ingest_meta(meta_path)
    if not docs_paths and not old_meta:
        notify("no_documents", data_dir=Path(data_dir), skipped=result.skipped_files)
        result.total_chunks = collection.count()
        return result

    cur_meta = build_ingest_meta(docs_paths)
    to_add, deleted, result.unchanged, result.reason = plan_changes(old_meta, cur_meta, collection.count())
    result.files_total = len(to_add) + len(deleted)
    notify(
        "plan",
        reason=result.reason,
        to_add=[cur_meta[fid]["name"] for fid in to_add],
        deleted=[old_meta[fid]["name"] for fid in deleted],
        unchanged=result.unchanged,
        skipped=result.skipped_files,
    )
    if not to_add and not deleted:
        notify("nochange")
        result.total_chunks = collection.count()
        return result

    # index_missing / legacy 都是整体重建：旧元数据作废，从空开始逐文件记录
    new_meta = {} if result.reason else dict(old_meta)
    try:
        if result.reason == "legacy":
            # 旧数据没有 fid，无法按文件删除。不能 delete_collection（运行中的服务持有该对象），按 ID 分批清空。
            old_ids = collection.get(include=[])["ids"]
            result.chunks_deleted += delete_chunk_ids(collection, old_ids, lock)
            result.changed = True
            notify("migrate", chunks=len(old_ids))

        for fid in deleted:
            name = old_meta[fid]["name"]
            count = delete_file_chunks(collection, fid=fid, lock=lock)
            result.chunks_deleted += count
            result.changed = result.changed or bool(count)
            new_meta.pop(fid, None)
            save_ingest_meta(new_meta, meta_path)
            result.files_done += 1
            notify("file_deleted", name=name, chunks_deleted=count)

        for fid in to_add:
            name = cur_meta[fid]["name"]
            notify("file_start", name=name)
            new_meta.pop(fid, None)  # 处理期间不认为它是最新的：失败时下次会重试
            result.changed = True
            try:
                added, stale, report = _index_file(data_dir / name, fid, collection, embed_fn, lock, batch_size)
            except Exception as exc:
                logger.exception("文件入库失败：%s", name)
                result.failed_files[name] = f"{type(exc).__name__}: {exc}"
                notify("file_failed", name=name, error=result.failed_files[name])
            else:
                result.chunks_added += added
                result.chunks_deleted += stale
                new_meta[fid] = cur_meta[fid]
                save_ingest_meta(new_meta, meta_path)
                notify("file_done", name=name, chunks_added=added, chunks_deleted=stale, report=report)
            result.files_done += 1
    finally:
        save_ingest_meta(new_meta, meta_path)  # 异常 / 中断时也把已完成的部分记下来

    result.total_chunks = collection.count()
    return result


def delete_document(
    name: str,
    data_dir: Path,
    collection,
    meta_path: Path,
    lock=None,
    sources: list[str] | None = None,
) -> int:
    """删除一个文档：源文件、索引中的片段（按 fid；旧版片段按 source 原始值）和元数据条目。返回删除的片段数。"""
    fid = file_id(Path(name))
    count = delete_file_chunks(collection, fid=fid, sources=sources, lock=lock)
    (Path(data_dir) / name).unlink(missing_ok=True)
    meta = load_ingest_meta(meta_path)
    if meta.pop(fid, None) is not None:
        save_ingest_meta(meta, meta_path)
    return count


def print_progress(event: str, **info) -> None:
    """命令行用的进度回调：把事件打印成与原脚本一致的中文提示。"""
    if event == "no_documents":
        for name in info["skipped"]:
            print(f"警告：跳过旧版 .doc：{name}，请另存为 .docx")
        print(f"错误：在 {info['data_dir']} 下没找到文档")
    elif event == "plan":
        for name in info["skipped"]:
            print(f"警告：跳过旧版 .doc：{name}，请另存为 .docx")
        if info["reason"] == "index_missing":
            print("检测到索引缺失或为空，将全量重建。")
        elif info["reason"] == "legacy":
            print("检测到旧版索引（缺少文件元数据），本次将迁移为可增量更新的索引。")
        print(
            f"文件状态：新增/修改 {len(info['to_add'])} 个，未变 {info['unchanged']} 个，"
            f"已删除 {len(info['deleted'])} 个"
        )
    elif event == "nochange":
        print("没有变化，无需更新。")
    elif event == "migrate":
        print(f"  已清空旧版片段 {info['chunks']} 个")
    elif event == "file_deleted":
        print(f"  已删除文件：{info['name']}")
    elif event == "file_start":
        print(f"索引中：{info['name']}")
    elif event == "file_done":
        report = info["report"]
        cleaning, splitting = report["cleaning"], report["splitting"]
        print(
            f"  -> {info['chunks_added']} 个片段 | "
            f"清除模板行 {cleaning['repeated_boilerplate_lines_removed']} | "
            f"空页 {cleaning['empty_documents_removed']} | "
            f"重复片段 {splitting['duplicate_chunks_removed']} | "
            f"带章节标题 {report['heading_chunks']}"
        )
        if report["parsed_documents"] and not info["chunks_added"]:
            print("  警告：文档没有可索引文本，扫描版 PDF 需要 OCR")
    elif event == "file_failed":
        print(f"  失败：{info['name']}：{info['error']}")
