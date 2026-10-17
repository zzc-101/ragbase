"""ragbase 命令行入口：serve / ingest / ask / chat / eval 五个子命令。

- 子命令的重型依赖（torch、chromadb、ragas 等）都在各自的处理函数里才导入，所以 `ragbase --help` 很快。
- 数据、索引、.env 等默认位置都在项目主目录下：环境变量 RAG_HOME，未设置则为当前工作目录（见 ragbase.config）。
"""
import argparse
import json
import sys
import uuid

from ragbase import __version__
from ragbase.config import get_settings, load_env

LLM_MISSING_HINT = "错误：请先在 .env 里配置 RAG_LLM_API_KEY、RAG_LLM_MODEL 和 RAG_LLM_BASE_URL（旧的 ARK_* 变量同样有效）"
INDEX_MISSING_HINT = "错误：索引不存在或为空，请先运行：ragbase ingest --full"
MODEL_LOADING_HINT = "加载本地 embedding 与 reranker 模型..."


# ---------------------------------------------------------------- serve（原 python api.py）

def _cmd_serve(args) -> None:
    from ragbase.api.app import serve

    serve(host=args.host, port=args.port)


# ---------------------------------------------------------------- ingest（原 ingest_incremental.py / ingest.py）

class LazyEmbedder:
    """第一次真正需要向量化时才加载本地模型，没有变化时不必等模型加载。"""

    def __init__(self):
        self._embeddings = None

    def __call__(self, texts):
        if self._embeddings is None:
            from ragbase.models.embeddings import Qwen3Embeddings

            print("加载本地 embedding 模型 Qwen3-Embedding-0.6B...")
            self._embeddings = Qwen3Embeddings()
        return self._embeddings.embed_documents(texts)


def _cmd_ingest(args) -> None:
    if args.full:
        _ingest_full()
    else:
        _ingest_incremental()


def _ingest_incremental() -> None:
    """增量构建：只处理新增 / 修改的文档，删除已不存在的；首次运行（库为空）等价于全量。"""
    import chromadb

    from ragbase.ingest.common import COLLECTION
    from ragbase.ingest.service import print_progress, run_ingest

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


def _ingest_full() -> None:
    """全量重建：先清空旧索引再重建。

    警告：会删除整个 chroma_db 目录。HTTP 服务（ragbase serve）运行期间绝对不要运行它——
    服务进程还握着旧的索引文件；服务在线时请用 POST /documents、POST /ingest 增量更新。
    """
    import shutil

    import chromadb

    from ragbase.ingest.common import COLLECTION
    from ragbase.ingest.parser import list_documents, report_skipped
    from ragbase.ingest.service import print_progress, run_ingest

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
    from ragbase.models.embeddings import Qwen3Embeddings

    embeddings = Qwen3Embeddings()
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(COLLECTION)

    # 库已清空、无元数据，run_ingest 会把所有文件当作新增；每个文件成功后才写入其元数据
    result = run_ingest(data_dir, collection, embeddings.embed_documents, meta_file, progress=print_progress)

    print(f"\n完成！共写入 {result.chunks_added} 个片段，向量库已存到 {db_dir}")
    if result.failed_files:
        print(f"有 {len(result.failed_files)} 个文件入库失败（见上方提示），修复后可运行 ragbase ingest 重试。")
        raise SystemExit(1)
    print("现在可以运行：ragbase ask 开始问答")


# ---------------------------------------------------------------- ask（原 query.py）

def _cmd_ask(args) -> None:
    from ragbase.ingest.common import collection_count
    from ragbase.retrieval.hybrid import UnknownSourceError
    from ragbase.retrieval.labels import source_label
    from ragbase.retrieval.pipeline import ask, build_pipeline

    settings = get_settings()
    if not settings.llm_configured:
        print(LLM_MISSING_HINT)
        return
    if not collection_count(settings.db_dir):
        print(INDEX_MISSING_HINT)
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


# ---------------------------------------------------------------- chat（原 chat.py）

def _args_text(args: dict) -> str:
    return ", ".join(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in args.items())


def run_turn(service, session_id: str, message: str, sources: list[str] | None) -> None:
    """发送一条消息：实时打印工具调用和回答文本，结束后打印编号来源。出错只打印提示，会话保持可用（失败的那一轮已回滚）。"""
    from ragbase.agent.tools import ToolRetrievalError
    from ragbase.retrieval.labels import source_label
    from ragbase.retrieval.pipeline import GenerationError

    started, citations = False, []
    try:
        for event, data in service.stream(session_id, message, sources):
            if event == "tool_call":
                print(f"  [工具] {data['name']}({_args_text(data['args'])})")
            elif event == "token":
                if not started:
                    print("\n答：", end="")
                    started = True
                print(data["text"], end="", flush=True)
            elif event == "citations":
                citations = data["citations"]
        print()
    except GenerationError as exc:
        print(f"\n错误：{'大模型响应超时' if exc.timeout else '大模型调用失败'}，请稍后重试（{exc}）")
        return
    except ToolRetrievalError as exc:
        print(f"\n错误：{exc}")
        return
    if citations:
        print("\n来源：")
        for number, doc in citations:
            print(f"[{number}] {source_label(doc.metadata)}")


def _cmd_chat(args) -> None:
    from ragbase.agent.service import build_chat_service
    from ragbase.ingest.common import collection_count
    from ragbase.retrieval.hybrid import UnknownSourceError
    from ragbase.retrieval.pipeline import build_pipeline

    settings = get_settings()
    if not settings.llm_configured:
        print(LLM_MISSING_HINT)
        return
    if not collection_count(settings.db_dir):
        print(INDEX_MISSING_HINT)
        return

    print(MODEL_LOADING_HINT)
    pipeline, _ = build_pipeline(settings=settings, build_bm25=True)
    sources = list(dict.fromkeys(args.source)) or None
    if sources:
        try:
            pipeline.retriever.resolve_sources(sources)
        except UnknownSourceError as exc:
            print(f"错误：未知的书名 {exc.unknown}")
            print("可选：" + "、".join(exc.valid))
            return
    service = build_chat_service(pipeline, settings)

    scope = f"，默认限定：{'、'.join(sources)}" if sources else ""
    print(f"对话已就绪{scope}。输入问题开始；/new 新会话，exit 退出")
    session_id = uuid.uuid4().hex
    while True:
        try:
            message = input("\n问：").strip()
        except EOFError:
            break
        if message.lower() in ("exit", "quit", "q"):
            print("再见！")
            break
        if message == "/new":
            service.delete(session_id)
            session_id = uuid.uuid4().hex
            print("已开始新会话")
            continue
        if message:
            run_turn(service, session_id, message[:2000], sources)  # 与 POST /chat 的 message 上限一致


# ---------------------------------------------------------------- eval（原 evaluate.py）

def _cmd_eval(args) -> None:
    from ragbase.evaluation.runner import run

    run(args)


# ---------------------------------------------------------------- 参数解析

def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()  # 默认值来自配置（环境变量 / .env），命令行参数可覆盖
    parser = argparse.ArgumentParser(
        prog="ragbase",
        description="ragbase：本地 RAG 知识库（项目主目录由环境变量 RAG_HOME 指定，默认当前目录）",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="{serve,ingest,ask,chat,eval}", required=True)

    serve = sub.add_parser("serve", help="启动 HTTP 服务（含 Web 界面）", description="启动 HTTP 服务（单进程，不要多 worker）")
    serve.add_argument("--host", default=None, help=f"监听地址（默认 {settings.host}，可由 RAG_HOST 设置）")
    serve.add_argument("--port", type=int, default=None, help=f"监听端口（默认 {settings.port}，可由 RAG_PORT 设置）")
    serve.set_defaults(func=_cmd_serve)

    ingest = sub.add_parser(
        "ingest",
        help="构建向量索引（默认增量；--full 全量重建）",
        description=(
            "扫描 data/ 下的文档（PDF / Word(.docx) / Markdown / 纯文本）-> 切分 -> 向量化 -> 存入 ChromaDB。"
            "默认增量：只处理新增 / 修改的文档，删除已不存在的；首次运行等价于全量。"
            "服务（ragbase serve）运行时请用 POST /ingest，不要同时运行本命令。"
        ),
    )
    ingest.add_argument(
        "--full",
        action="store_true",
        help="全量重建：先删除整个 chroma_db 再重建（服务运行期间绝对不要使用）",
    )
    ingest.set_defaults(func=_cmd_ingest)

    ask = sub.add_parser("ask", help="命令行问答", description="命令行问答：输入问题 -> 检索相关片段 -> LLM 基于片段生成答案；输入 exit 退出")
    ask.add_argument(
        "--mode",
        choices=("vector", "bm25", "hybrid"),
        default=settings.mode,
        help=f"召回方式：向量 / BM25 / 两路 RRF 融合（默认 {settings.mode}，可由 RAG_MODE 设置）",
    )
    ask.add_argument(
        "--min-score",
        type=float,
        default=settings.min_score,
        help=f"最相关片段的精排得分（0~1）低于该值则拒答、不调用大模型（默认 {settings.min_score}；0 表示不拒答）",
    )
    ask.add_argument(
        "--context-min-score",
        type=float,
        default=settings.context_min_score,
        help=f"只把精排得分不低于该值的片段交给大模型（最相关片段总是保留；默认 {settings.context_min_score}；0 表示不过滤）",
    )
    ask.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME",
        help="只在这本书里检索（文件名，如 xxx.pdf；可重复指定多本）；不指定则检索全部",
    )
    ask.set_defaults(func=_cmd_ask)

    chat = sub.add_parser(
        "chat",
        help="命令行多轮对话（Agent）",
        description="命令行多轮对话：Agent 按需检索知识库并带 [n] 引用回答（与 POST /chat 同一套 Agent，进程内运行）。/new 开始新会话，exit 退出；会话历史只在本进程内存里。",
    )
    chat.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME",
        help="默认只在这本书里检索（文件名，可重复指定）；模型调用工具时没有另外指定书名就用它；不指定则检索全部",
    )
    chat.set_defaults(func=_cmd_chat)

    ev = sub.add_parser("eval", help="评估 RAG 检索与回答质量", description="运行 Ragas、词面检索指标和原有中文裁判的综合评测")
    ev.add_argument("--limit", type=int, help="只评估前 N 道题，适合冒烟测试")
    ev.add_argument("--start-id", type=int, default=1, help="从指定题号开始")
    ev.add_argument(
        "--skip-ragas",
        action="store_true",
        help="跳过 Ragas 指标，减少 API 调用",
    )
    ev.add_argument(
        "--skip-legacy-judge",
        action="store_true",
        help="跳过原有中文 LLM 裁判",
    )
    ev.add_argument(
        "--retrieval-mode",
        choices=("vector", "bm25", "hybrid"),
        default=settings.mode,
        help=f"召回方式：向量 / BM25 / 两路 RRF 融合（默认 {settings.mode}，可由 RAG_MODE 设置）",
    )
    ev.add_argument(
        "--retrieval-only",
        action="store_true",
        help="只评估召回与精排的词面指标，不调用 LLM / Ragas / 裁判（不产生 API 费用）",
    )
    ev.add_argument(
        "--min-score",
        type=float,
        default=settings.min_score,
        help=f"最相关片段的精排得分低于该值则拒答、不调用 LLM（默认 {settings.min_score}；0 表示不拒答）",
    )
    ev.add_argument(
        "--context-min-score",
        type=float,
        default=settings.context_min_score,
        help=f"只把精排得分不低于该值的片段交给 LLM（top1 总是保留；默认 {settings.context_min_score}；0 表示不过滤）",
    )
    ev.set_defaults(func=_cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env()  # 载入主目录下的 .env（Settings 也会读它；这里让 ARK_* 等变量进入 os.environ）
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
