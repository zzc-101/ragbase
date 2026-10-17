"""命令行问答：输入问题 -> 检索相关片段 -> LLM 基于片段生成答案。

运行：uv run query.py [--mode {vector,bm25,hybrid}]
输入问题即可，输入 exit 退出。
"""
import argparse
import os

from dotenv import load_dotenv
from config import get_settings
from hybrid_retriever import UnknownSourceError
from rag_common import collection_count
from rag_pipeline import ask, build_pipeline

load_dotenv()


def parse_args():
    settings = get_settings()  # 默认值来自配置（环境变量 / .env），命令行参数可覆盖
    parser = argparse.ArgumentParser(description="RAG 命令行问答")
    parser.add_argument(
        "--mode",
        choices=("vector", "bm25", "hybrid"),
        default=settings.mode,
        help=f"召回方式：向量 / BM25 / 两路 RRF 融合（默认 {settings.mode}，可由 RAG_MODE 设置）",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=settings.min_score,
        help=f"最相关片段的精排得分（0~1）低于该值则拒答、不调用大模型（默认 {settings.min_score}；0 表示不拒答）",
    )
    parser.add_argument(
        "--context-min-score",
        type=float,
        default=settings.context_min_score,
        help=f"只把精排得分不低于该值的片段交给大模型（最相关片段总是保留；默认 {settings.context_min_score}；0 表示不过滤）",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME",
        help="只在这本书里检索（文件名，如 xxx.pdf；可重复指定多本）；不指定则检索全部",
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
    """引用来源：文件名[:第N页][「末级章节」][召回路径[|精排得分]]；仅在元数据有 page（PDF）时显示页码，有 heading_path 时显示章节。"""
    name = os.path.basename(metadata.get("source", "?"))
    page = metadata.get("page")
    page_text = f":第{page + 1}页" if isinstance(page, int) else ""
    heading = metadata.get("heading_path")
    heading_text = f"「{heading.split(' > ')[-1]}」" if heading else ""  # 只显示最末一级，保持简短
    score = metadata.get("rerank_score")
    score_text = f"|{score:.2f}" if isinstance(score, float) else ""
    return f"{name}{page_text}{heading_text}[{route_label(metadata)}{score_text}]"


def main():
    args = parse_args()
    settings = get_settings()
    if not settings.llm_configured:
        print("错误：请先在 .env 里配置 RAG_LLM_API_KEY、RAG_LLM_MODEL 和 RAG_LLM_BASE_URL（旧的 ARK_* 变量同样有效）")
        return
    if not collection_count(settings.db_dir):
        print("错误：索引不存在或为空，请先运行：uv run ingest.py")
        return

    print("加载本地 embedding 与 reranker 模型...")
    pipeline, _ = build_pipeline(mode=args.mode)
    sources = list(dict.fromkeys(args.source)) or None
    if sources:
        try:
            pipeline.retriever.resolve_sources(sources)
        except UnknownSourceError as exc:
            print(f"错误：未知的书名 {exc.unknown}")
            print("可选：" + "、".join(exc.valid))
            return

    scope = f"，限定：{'、'.join(sources)}" if sources else ""
    print(f"RAG 问答已就绪（召回模式：{args.mode}{scope}），输入问题开始（输入 exit 退出）")
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
            rerank_k=settings.rerank_k,
            min_score=args.min_score,
            context_min_score=args.context_min_score,
            sources=sources,
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
