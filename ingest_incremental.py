"""增量构建向量索引：只处理有变化的文档（新增/修改），删除已不存在的。

运行：uv run ingest_incremental.py
首次运行等价于全量（库为空）。后续新增/修改文档后运行，只索引变化部分，快很多。
支持格式：PDF / Word(.docx) / Markdown / 纯文本
"""
from langchain_community.vectorstores import Chroma
import chromadb
from document_parser import list_documents, report_skipped
from embed import Qwen3Embeddings
from rag_common import (
    DATA_DIR, DB_DIR, COLLECTION,
    build_ingest_meta, collection_count, load_ingest_meta,
    prepare_file_chunks, save_ingest_meta,
)


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


def main():
    docs_paths, skipped = list_documents(DATA_DIR)
    report_skipped(skipped)
    old_meta = load_ingest_meta()
    if not docs_paths and not old_meta:
        print(f"错误：在 {DATA_DIR} 下没找到文档")
        return

    cur_meta = build_ingest_meta(docs_paths)
    to_add, deleted, unchanged, reason = plan_changes(
        old_meta, cur_meta, collection_count(),
    )
    if reason == "index_missing":
        print("检测到索引缺失或为空，将全量重建。")
    elif reason == "legacy":
        print("检测到旧版索引（缺少文件元数据），本次将迁移为可增量更新的索引。")

    print(f"文件状态：新增/修改 {len(to_add)} 个，未变 {unchanged} 个，已删除 {len(deleted)} 个")

    if not to_add and not deleted:
        print("没有变化，无需更新。")
        return

    print("加载本地 embedding 模型 Qwen3-Embedding-0.6B...")
    embeddings = Qwen3Embeddings()
    client = chromadb.PersistentClient(path=DB_DIR)
    if reason == "legacy":
        # 旧数据没有 fid，无法安全地按文件删除，重建集合。
        client.delete_collection(COLLECTION)

    vectordb = Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )
    collection = vectordb._collection

    # 1. 删除已移除文件的旧片段（用 fid 前缀过滤）
    for fid in deleted:
        name = old_meta[fid]["name"]
        collection.delete(where={"fid": fid})
        print(f"  已删除文件：{name}")

    # 2. 对新增/修改的文件：先删该文件旧片段，再重新索引
    for fid in to_add:
        info = cur_meta[fid]
        path = DATA_DIR / info["name"]

        # 内容变了的话，先清掉该文件旧片段
        collection.delete(where={"fid": fid})

        print(f"索引中：{info['name']}")
        texts, metadatas, ids, report = prepare_file_chunks(path)

        # 必须通过 vectordb 写入，使其使用 Qwen3Embeddings 生成 1024 维向量。
        batch_size = 1000
        for i in range(0, len(texts), batch_size):
            vectordb.add_texts(
                texts=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size],
            )
        cleaning = report["cleaning"]
        splitting = report["splitting"]
        print(
            f"  -> {len(texts)} 个片段 | "
            f"清除模板行 {cleaning['repeated_boilerplate_lines_removed']} | "
            f"空页 {cleaning['empty_documents_removed']} | "
            f"重复片段 {splitting['duplicate_chunks_removed']}"
        )
        if report["parsed_documents"] and not texts:
            print("  警告：文档没有可索引文本，扫描版 PDF 需要 OCR")

    # 3. 保存新指纹
    save_ingest_meta(cur_meta)
    total = collection.count()
    print(f"\n完成！当前向量库共 {total} 个片段，存于 {DB_DIR}")


if __name__ == "__main__":
    main()
