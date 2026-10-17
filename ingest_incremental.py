"""增量构建向量索引：只处理有变化的文档（新增/修改），删除已不存在的。

运行：uv run ingest_incremental.py
首次运行等价于全量（库为空）。后续新增/修改文档后运行，只索引变化部分，快很多。
支持格式：PDF / Word(.docx) / Markdown / 纯文本
入库逻辑在 ingest_service.py（与 HTTP 服务的 POST /ingest 共用）。
注意：服务正在运行时请用 POST /ingest，不要同时运行本脚本。
"""
import chromadb

from config import get_settings
from ingest_service import plan_changes, print_progress, run_ingest  # noqa: F401  (plan_changes 保留导出)
from rag_common import COLLECTION


class LazyEmbedder:
    """第一次真正需要向量化时才加载本地模型，没有变化时不必等模型加载。"""

    def __init__(self):
        self._embeddings = None

    def __call__(self, texts):
        if self._embeddings is None:
            from embed import Qwen3Embeddings

            print("加载本地 embedding 模型 Qwen3-Embedding-0.6B...")
            self._embeddings = Qwen3Embeddings()
        return self._embeddings.embed_documents(texts)


def main():
    settings = get_settings()
    client = chromadb.PersistentClient(path=str(settings.db_dir))
    collection = client.get_or_create_collection(COLLECTION)
    result = run_ingest(
        settings.data_dir,
        collection,
        LazyEmbedder(),
        settings.meta_file,
        progress=print_progress,
    )
    if result.failed_files:
        print(f"\n有 {len(result.failed_files)} 个文件入库失败，其余文件已写入；修复后重新运行即可重试。")
    if result.changed:
        print(f"\n完成！当前向量库共 {result.total_chunks} 个片段，存于 {settings.db_dir}")
    if result.failed_files:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
