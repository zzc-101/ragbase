"""构建向量索引：扫描 data/ 下的文档 -> 切分 -> 向量化 -> 存入 ChromaDB。

运行：uv run ingest.py
可重复运行，每次会先清空旧索引重建。
支持格式：PDF / Word(.docx) / Markdown / 纯文本

警告：本脚本会删除整个 chroma_db 目录。HTTP 服务（api.py）运行期间绝对不要运行它——
服务进程还握着旧的索引文件；服务在线时请用 POST /documents、POST /ingest 增量更新。
"""
import shutil

import chromadb

from config import get_settings
from document_parser import list_documents, report_skipped
from ingest_service import print_progress, run_ingest
from rag_common import COLLECTION


def main():
    settings = get_settings()
    data_dir, db_dir, meta_file = settings.data_dir, settings.db_dir, settings.meta_file
    docs_paths, skipped = list_documents(data_dir)
    if not docs_paths:
        report_skipped(skipped)
        print(f"错误：在 {data_dir} 下没找到文档")
        print("支持格式：PDF / Word(.docx) / Markdown / 纯文本")
        return

    # 清空旧索引后重建
    if db_dir.exists():
        shutil.rmtree(db_dir)
    # 避免重建中断后留下与半成品索引不匹配的旧元数据
    meta_file.unlink(missing_ok=True)

    # 本地 embedding：Qwen3-Embedding-0.6B（2025年，100+语言含中文+代码）
    print("加载本地 embedding 模型 Qwen3-Embedding-0.6B...")
    from embed import Qwen3Embeddings

    embeddings = Qwen3Embeddings()
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(COLLECTION)

    # 库已清空、无元数据，run_ingest 会把所有文件当作新增；每个文件成功后才写入其元数据
    result = run_ingest(data_dir, collection, embeddings.embed_documents, meta_file, progress=print_progress)

    print(f"\n完成！共写入 {result.chunks_added} 个片段，向量库已存到 {db_dir}")
    if result.failed_files:
        print(f"有 {len(result.failed_files)} 个文件入库失败（见上方提示），修复后可运行 ingest_incremental.py 重试。")
        raise SystemExit(1)
    print("现在可以运行：uv run query.py 开始问答")


if __name__ == "__main__":
    main()
