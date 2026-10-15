"""命令行问答：输入问题 -> 检索相关片段 -> LLM 基于片段生成答案。

运行：uv run query.py [--mode {vector,bm25,hybrid}]
输入问题即可，输入 exit 退出。
"""
import argparse
import os

from dotenv import load_dotenv
from rag_common import collection_count
from rag_pipeline import CONTEXT_MIN_SCORE, RERANK_MIN_SCORE, ask, build_pipeline

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="RAG 命令行问答")
    parser.add_argument(
        "--mode",
        choices=("vector", "bm25", "hybrid"),
        default="hybrid",
        help="召回方式：向量 / BM25 / 两路 RRF 融合（默认）",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=RERANK_MIN_SCORE,
        help=f"最相关片段的精排得分（0~1）低于该值则拒答、不调用大模型（默认 {RERANK_MIN_SCORE}；0 表示不拒答）",
    )
    parser.add_argument(
        "--context-min-score",
        type=float,
        default=CONTEXT_MIN_SCORE,
        help=f"只把精排得分不低于该值的片段交给大模型（最相关片段总是保留；默认 {CONTEXT_MIN_SCORE}；0 表示不过滤）",
    )
    return parser.parse_args()


def route_label(metadata: dict) -> str:
    """召回来源：V=向量、B=BM25，数字为该路内排名，如 V3 B1。"""
    parts = []
    if metadata.get("vec_rank"):
        parts.append(f"V{metadata['vec_rank']}")
    if metadata.get("bm25_rank"):
        parts.append(f"B{metadata['bm25_rank']}")
    return "+".join(parts) or "?"


def source_label(metadata: dict) -> str:
    """引用来源：文件名[:第N页][召回路径[|精排得分]]；仅在元数据有 page（PDF）时显示页码。"""
    name = os.path.basename(metadata.get("source", "?"))
    page = metadata.get("page")
    page_text = f":第{page + 1}页" if isinstance(page, int) else ""
    score = metadata.get("rerank_score")
    score_text = f"|{score:.2f}" if isinstance(score, float) else ""
    return f"{name}{page_text}[{route_label(metadata)}{score_text}]"


def main():
    args = parse_args()
    if not os.getenv("ARK_CHAT_API_KEY"):
        print("错误：请先在 .env 文件里填写 ARK_CHAT_API_KEY")
        return
    if not collection_count():
        print("错误：索引不存在或为空，请先运行：uv run ingest.py")
        return

    print("加载本地 embedding 与 reranker 模型...")
    pipeline, _ = build_pipeline(retrieve_k=20, mode=args.mode)

    print(f"RAG 问答已就绪（召回模式：{args.mode}），输入问题开始（输入 exit 退出）")
    while True:
        q = input("\n问：").strip()
        if q.lower() in ("exit", "quit", "q"):
            print("再见！")
            break
        if not q:
            continue
        result = ask(
            pipeline,
            q,
            rerank_k=5,
            min_score=args.min_score,
            context_min_score=args.context_min_score,
        )

        if result.refused:
            print(f"\n答：{result.answer}")
            if result.top_score is not None:
                print(
                    f"（最相关片段得分 {result.top_score:.2f}，低于阈值 {args.min_score:.2f}，"
                    "未调用大模型）"
                )
            continue

        if result.evidence:
            print(f"\n依据：\n{result.evidence}")
        print(f"\n答：{result.answer}")

        # 打印引用来源：编号与大模型看到的 [片段N] 一一对应
        if result.docs:
            print("\n来源：")
            for number, doc in enumerate(result.docs, 1):
                print(f"[片段{number}] {source_label(doc.metadata)}")


if __name__ == "__main__":
    main()
