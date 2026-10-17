"""ragbase 命令行入口：init / models / doctor / serve / ingest / ask / chat / eval / migrate-store 子命令。

- 子命令的重型依赖（torch、chromadb、qdrant-client、ragas 等）都在各自的处理函数里才导入，所以 `ragbase --help` 很快。
- 数据、索引、.env 等默认位置都在项目主目录下：环境变量 RAG_HOME，未设置则为当前工作目录（见 ragbase.config）。
"""
import argparse
import json
import sys
import uuid

from ragbase import __version__
from ragbase.config import VECTOR_STORES, get_settings, load_env

LLM_MISSING_HINT = "错误：请先在 .env 里配置 RAG_LLM_API_KEY、RAG_LLM_MODEL 和 RAG_LLM_BASE_URL（旧的 ARK_* 变量同样有效）"
INDEX_MISSING_HINT = "错误：索引不存在或为空，请先运行：ragbase ingest --full"
MODEL_LOADING_HINT = "加载本地 embedding 与 reranker 模型..."


# ---------------------------------------------------------------- init / models / doctor（开箱即用）

def _cmd_init(args) -> None:
    from pathlib import Path

    from ragbase.config import project_home
    from ragbase.scaffold import generate_api_key, init_home

    home = Path(args.dir).expanduser().resolve() if args.dir else project_home()
    api_key = generate_api_key() if args.api_key else None
    actions = init_home(home, force=args.force, api_key=api_key)

    labels = {"created": "创建", "skipped": "跳过", "overwritten": "覆盖"}
    print(f"项目主目录：{home}")
    for action in actions:
        note = f"（{action.note}）" if action.note else ""
        print(f"  [{labels[action.status]}] {action.path.relative_to(home)}{note}")
    env_status = next(a.status for a in actions if a.path.name == ".env")
    if api_key:
        if env_status == "skipped":
            print("提示：.env 已存在，没有写入新密钥；要重新生成请加 --force（旧 .env 会备份为 .env.bak），或自己在 .env 里设置 RAG_API_KEY。")
        else:
            print("已生成随机接口密钥并写入 .env 的 RAG_API_KEY；Web 界面弹窗里输入它（grep RAG_API_KEY .env 可查看）。")

    in_home = home == Path.cwd().resolve()
    print("\n下一步：")
    step = 1
    if not in_home:
        print(f"  {step}. 进入该目录（或设置环境变量 RAG_HOME）：cd {home}   # 或 export RAG_HOME={home}")
        step += 1
    print(f"  {step}. 编辑 .env，填入大模型配置 RAG_LLM_BASE_URL / RAG_LLM_API_KEY / RAG_LLM_MODEL（任意 OpenAI 兼容服务）")
    if not api_key:
        print("     对外开放服务（RAG_HOST=0.0.0.0 / Docker）前，务必同时设置 RAG_API_KEY（可用 ragbase init --api-key --force 生成）")
    print(f"  {step + 1}. 下载本地模型：ragbase models download   （国内网络：HF_ENDPOINT=https://hf-mirror.com ragbase models download）")
    print(f"  {step + 2}. 把文档放进 data/，然后建立索引：ragbase ingest")
    print(f"  {step + 3}. 体检：ragbase doctor；启动：ragbase serve   （浏览器打开 http://127.0.0.1:8000/）")


def _cmd_models_download(args) -> None:
    from ragbase.models.download import ensure_models

    settings = get_settings()
    print("检查本地模型缓存（已缓存的会跳过，不联网）...")
    results = ensure_models(settings)
    failed = [spec for spec, status in results if status.startswith("failed")]
    downloaded = sum(status == "downloaded" for _, status in results)
    cached = sum(status == "cached" for _, status in results)
    print(f"\n汇总：{cached} 个已缓存，{downloaded} 个新下载，{len(failed)} 个失败")
    if failed:
        print("下载失败：" + "、".join(spec.repo_id for spec in failed))
        print("排查：1) 检查网络；2) 国内网络设置镜像后重试：HF_ENDPOINT=https://hf-mirror.com ragbase models download；"
              "3) 可重复运行，已下好的文件不会重下（断点续传）")
        raise SystemExit(1)
    print("模型就绪。")


def _cmd_doctor(args) -> None:
    from ragbase.doctor import run_doctor

    code = run_doctor(get_settings(), check_llm_call=args.check_llm)
    if code:
        raise SystemExit(code)


# ---------------------------------------------------------------- serve（原 python api.py）

def _ensure_models_for_serve(settings, download: bool) -> None:
    """启动服务前确认本地模型已缓存：缺失时给出明确提示后退出，而不是在加载模型时抛出一长串 traceback。

    --download-models：缺什么就先下载什么（Docker 首次启动用）。没有该参数时：索引为空的服务本来就不加载模型，直接放行；
    索引非空而模型缺失才拦下。
    """
    from ragbase.models.download import ensure_models, missing_models

    if download:
        results = ensure_models(settings)
        if any(status.startswith("failed") for _, status in results):
            print("错误：模型下载失败，无法启动。国内网络可设置 HF_ENDPOINT=https://hf-mirror.com 后重试。")
            raise SystemExit(1)
        return
    missing = missing_models(settings)
    if not missing:
        return
    from ragbase.ingest.common import collection_count

    if not collection_count(settings):
        return  # 索引为空：服务不加载模型也能启动（只是还不能问答），先让用户看到界面
    print("错误：本地模型尚未下载，服务无法加载：")
    for spec in missing:
        print(f"  - {spec.role}：{spec.repo_id}")
    print("请先运行：ragbase models download   （国内网络：HF_ENDPOINT=https://hf-mirror.com ragbase models download）")
    print("或者让 serve 自动下载：ragbase serve --download-models")
    raise SystemExit(1)


def _cmd_serve(args) -> None:
    _ensure_models_for_serve(get_settings(), args.download_models)
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

            print(f"加载本地 embedding 模型 {get_settings().embedding_model}...")
            self._embeddings = Qwen3Embeddings()
        return self._embeddings.embed_documents(texts)


def _cmd_ingest(args) -> None:
    if args.full:
        _ingest_full()
    else:
        _ingest_incremental()


def _ingest_incremental() -> None:
    """增量构建：只处理新增 / 修改的文档，删除已不存在的；首次运行（库为空）等价于全量。"""
    from ragbase.ingest.service import print_progress, run_ingest
    from ragbase.vectorstores import create_vector_store

    settings = get_settings()
    store = create_vector_store(settings)
    try:
        result = run_ingest(
            settings.data_dir,
            store,
            LazyEmbedder(),
            settings.meta_file,
            progress=print_progress,
        )
        if result.failed_files:
            print(f"\n有 {len(result.failed_files)} 个文件入库失败，其余文件已写入；修复后重新运行即可重试。")
        if result.changed:
            print(f"\n完成！当前向量库共 {result.total_chunks} 个片段，存于 {store.location}（{store.name}）")
    finally:
        store.close()
    if result.failed_files:
        raise SystemExit(1)


def _ingest_full() -> None:
    """全量重建：先清空旧索引再重建。

    警告：会清空整个向量库（Chroma 集合 / Qdrant 集合）。HTTP 服务（ragbase serve）运行期间绝对不要运行它——
    服务进程还握着旧的索引；服务在线时请用 POST /documents、POST /ingest 增量更新。
    """
    from ragbase.ingest.parser import list_documents, report_skipped
    from ragbase.ingest.service import print_progress, run_ingest
    from ragbase.vectorstores import create_vector_store

    settings = get_settings()
    data_dir, meta_file = settings.data_dir, settings.meta_file
    docs_paths, skipped = list_documents(data_dir)
    if not docs_paths:
        report_skipped(skipped)
        print(f"错误：在 {data_dir} 下没找到文档")
        print("支持格式：PDF / Word(.docx) / Markdown / 纯文本")
        return

    store = create_vector_store(settings)
    try:
        # 清空旧索引后重建
        store.reset()
        # 避免重建中断后留下与半成品索引不匹配的旧元数据
        meta_file.unlink(missing_ok=True)

        # 本地 embedding：默认 Qwen3-Embedding-0.6B（2025年，100+语言含中文+代码），可用 RAG_EMBEDDING_MODEL 更换
        print(f"加载本地 embedding 模型 {settings.embedding_model}...")
        from ragbase.models.embeddings import Qwen3Embeddings

        embeddings = Qwen3Embeddings()
        # 库已清空、无元数据，run_ingest 会把所有文件当作新增；每个文件成功后才写入其元数据
        result = run_ingest(data_dir, store, embeddings.embed_documents, meta_file, progress=print_progress)
    finally:
        store.close()

    print(f"\n完成！共写入 {result.chunks_added} 个片段，向量库（{store.name}）已存到 {store.location}")
    if result.failed_files:
        print(f"有 {len(result.failed_files)} 个文件入库失败（见上方提示），修复后可运行 ragbase ingest 重试。")
        raise SystemExit(1)
    print("现在可以运行：ragbase ask 开始问答")


# ---------------------------------------------------------------- migrate-store（向量库后端间迁移）

def _cmd_migrate_store(args) -> None:
    from ragbase.vectorstores import create_vector_store
    from ragbase.vectorstores.migrate import MigrationError, migrate_store

    settings = get_settings()
    target_name = args.to
    source_name = args.from_ or next(name for name in VECTOR_STORES if name != target_name)
    if source_name == target_name:
        print(f"错误：源库与目标库相同（{source_name}）")
        raise SystemExit(2)
    source = create_vector_store(settings, backend=source_name)
    target = create_vector_store(settings, backend=target_name)
    print(f"迁移：{source_name}（{source.location}）-> {target_name}（{target.location}），集合 {settings.collection}")
    print("提示：服务（ragbase serve）运行期间不要迁移；迁移不重新向量化，片段 ID 与正文不变，.ingest_meta.json 和 BM25 缓存继续有效。")
    try:
        result = migrate_store(
            source,
            target,
            batch_size=args.batch_size,
            force=args.force,
            progress=lambda done, total: print(f"  已复制 {done}/{total}"),
        )
    except MigrationError as exc:
        print(f"错误：{exc}")
        raise SystemExit(1)
    finally:
        source.close()
        target.close()
    print(f"完成：源库 {result.source_count} 个片段，已复制 {result.copied}，目标库现有 {result.target_count}。")
    print(f"现在把 RAG_VECTOR_STORE={target_name} 写进 .env 并重启服务即可使用新的向量库。")


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
    if not collection_count(settings):
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
    if not collection_count(settings):
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
    sub = parser.add_subparsers(dest="command", metavar="{init,models,doctor,serve,ingest,ask,chat,eval,migrate-store}", required=True)

    init = sub.add_parser(
        "init",
        help="初始化项目主目录（data/、.env、bm25_userdict.txt）",
        description="在项目主目录生成 data/（含 README.txt）、.env（取自内置模板，全是注释掉的示例）和默认 bm25_userdict.txt。默认绝不覆盖已有文件。",
    )
    init.add_argument("dir", nargs="?", default=None, help="项目主目录（默认：环境变量 RAG_HOME，未设置则为当前目录）")
    init.add_argument("--force", action="store_true", help="覆盖已有文件（旧文件先备份为 <名字>.bak）")
    init.add_argument("--api-key", action="store_true", help="生成一把随机接口密钥写入新 .env 的 RAG_API_KEY（对外开放服务时用）")
    init.set_defaults(func=_cmd_init)

    models = sub.add_parser("models", help="管理本地模型（embedding + reranker）", description="管理本地模型。缓存位置跟随 HF_HOME，镜像用 HF_ENDPOINT。")
    models_sub = models.add_subparsers(dest="models_command", metavar="{download}", required=True)
    download = models_sub.add_parser(
        "download",
        help="下载 embedding 与 reranker 权重到 Hugging Face 缓存",
        description=(
            f"下载 {settings.embedding_model}（RAG_EMBEDDING_MODEL）与 {settings.reranker_model}（RAG_RERANKER_MODEL）到 Hugging Face 缓存"
            "（HF_HOME，默认 ~/.cache/huggingface）；已缓存的跳过。国内网络：HF_ENDPOINT=https://hf-mirror.com ragbase models download"
        ),
    )
    download.set_defaults(func=_cmd_models_download)

    doctor = sub.add_parser(
        "doctor",
        help="环境自检（Python、路径、设备、模型、向量库、大模型配置、鉴权、Web 资源）",
        description="一屏环境自检，给出 ✓ / ! / ✗ 和修复提示；有 ✗ 时退出码非零。默认不调用大模型、不加载模型。",
    )
    doctor.add_argument("--check-llm", action="store_true", help="额外发一次最小的大模型请求，验证地址 / 密钥 / 模型名（会产生极少量 API 调用）")
    doctor.set_defaults(func=_cmd_doctor)

    serve = sub.add_parser("serve", help="启动 HTTP 服务（含 Web 界面）", description="启动 HTTP 服务（单进程，不要多 worker）")
    serve.add_argument("--host", default=None, help=f"监听地址（默认 {settings.host}，可由 RAG_HOST 设置）")
    serve.add_argument("--port", type=int, default=None, help=f"监听端口（默认 {settings.port}，可由 RAG_PORT 设置）")
    serve.add_argument(
        "--download-models",
        action="store_true",
        help="启动前自动下载缺失的本地模型（默认不下载：模型缺失时提示先运行 ragbase models download）",
    )
    serve.set_defaults(func=_cmd_serve)

    ingest = sub.add_parser(
        "ingest",
        help="构建向量索引（默认增量；--full 全量重建）",
        description=(
            "扫描 data/ 下的文档（PDF / Word(.docx) / Markdown / 纯文本）-> 切分 -> 向量化 -> 存入向量库（RAG_VECTOR_STORE：chroma 或 qdrant）。"
            "默认增量：只处理新增 / 修改的文档，删除已不存在的；首次运行等价于全量。"
            "服务（ragbase serve）运行时请用 POST /ingest，不要同时运行本命令。"
        ),
    )
    ingest.add_argument(
        "--full",
        action="store_true",
        help="全量重建：先清空整个向量库再重建（服务运行期间绝对不要使用）",
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

    migrate = sub.add_parser(
        "migrate-store",
        help="把向量库从一个后端迁移到另一个（不重新向量化）",
        description=(
            "把 ID、向量、正文和元数据按批从源向量库复制到目标向量库，不重新向量化，所以很快且不需要加载模型。"
            "两个后端各自的位置取自配置（Chroma：RAG_DB_DIR；Qdrant：RAG_QDRANT_PATH 或 RAG_QDRANT_URL），集合名取 RAG_COLLECTION。"
            "目标库非空时拒绝，加 --force 会先清空目标库。服务运行期间不要迁移。"
        ),
    )
    migrate.add_argument("--to", required=True, choices=VECTOR_STORES, help="目标后端")
    migrate.add_argument("--from", dest="from_", choices=VECTOR_STORES, default=None, help="源后端（默认：另一个后端）")
    migrate.add_argument("--force", action="store_true", help="目标库非空时先清空它再复制")
    migrate.add_argument("--batch-size", type=int, default=500, help="每批复制的片段数（默认 500）")
    migrate.set_defaults(func=_cmd_migrate_store)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env()  # 载入主目录下的 .env（Settings 也会读它；这里让 ARK_* 等变量进入 os.environ）
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
