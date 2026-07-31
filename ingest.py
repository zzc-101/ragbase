"""构建向量索引：扫描 data/ 下的文档 -> 切分 -> 向量化 -> 存入 ChromaDB。

运行：uv run ingest.py
可重复运行，每次会先清空旧索引重建。
支持格式：PDF / Word(.docx) / Markdown / 纯文本
"""
import shutil
from pathlib import Path

from langchain_community.vectorstores import Chroma
import chromadb
from embed import Qwen3Embeddings
from rag_common import (
    DATA_DIR, DB_DIR, COLLECTION, META_FILE,
    build_ingest_meta, list_documents, prepare_file_chunks, save_ingest_meta,
)


def main():
    docs_paths = list_documents()
    if not docs_paths:
        print(f"错误：在 {DATA_DIR} 下没找到文档")
        print(f"支持格式：PDF / Word(.docx) / Markdown / 纯文本")
        return

    # 清空旧索引后重建
    if Path(DB_DIR).exists():
        shutil.rmtree(DB_DIR)
    # 避免重建中断后留下与半成品索引不匹配的旧元数据
    META_FILE.unlink(missing_ok=True)

    # 本地 embedding：Qwen3-Embedding-0.6B（2025年，100+语言含中文+代码）
    print("加载本地 embedding 模型 Qwen3-Embedding-0.6B...")
    embeddings = Qwen3Embeddings()
    client = chromadb.PersistentClient(path=DB_DIR)
    vectordb = Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )

    # 逐文件解析，并以稳定 ID 分批写入
    batch_size = 1000
    total = 0
    for file_index, path in enumerate(docs_paths, 1):
        print(f"  [{file_index}/{len(docs_paths)}] 解析并索引：{path.name}")
        texts, metadatas, ids = prepare_file_chunks(path)
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            vectordb.add_texts(
                texts=texts[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
        total += len(texts)
        print(f"    -> {len(texts)} 个片段")

    # 只有全部文件成功入库后才更新元数据
    save_ingest_meta(build_ingest_meta(docs_paths))

    print(f"\n完成！共写入 {total} 个片段，向量库已存到 {DB_DIR}")
    print("现在可以运行：uv run query.py 开始问答")


if __name__ == "__main__":
    main()
