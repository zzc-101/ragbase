"""命令行问答：输入问题 -> 检索相关片段 -> LLM 基于片段生成答案。

运行：uv run query.py
输入问题即可，输入 exit 退出。
"""
import os

from dotenv import load_dotenv
from rag_common import DB_DIR
from rag_pipeline import ask, build_pipeline

load_dotenv()


def main():
    if not os.getenv("ARK_CHAT_API_KEY"):
        print("错误：请先在 .env 文件里填写 ARK_CHAT_API_KEY")
        return
    if not os.path.exists(DB_DIR):
        print("错误：还没建立索引，请先运行：uv run ingest.py")
        return

    print("加载本地 embedding 与 reranker 模型...")
    pipeline, _ = build_pipeline(retrieve_k=20)

    print("RAG 问答已就绪，输入问题开始（输入 exit 退出）")
    while True:
        q = input("\n问：").strip()
        if q.lower() in ("exit", "quit", "q"):
            print("再见！")
            break
        if not q:
            continue
        answer, docs, _ = ask(pipeline, q, rerank_k=5)

        print(f"\n答：{answer}")

        # 打印引用来源（文件名 + 页码）
        sources = set()
        for doc in docs:
            src = os.path.basename(doc.metadata.get("source", "?"))
            page = doc.metadata.get("page", 0)
            sources.add(f"{src}:第{page + 1}页")
        if sources:
            print("来源：" + " | ".join(sorted(sources)))


if __name__ == "__main__":
    main()
