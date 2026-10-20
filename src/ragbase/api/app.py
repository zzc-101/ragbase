"""RAG HTTP 服务（FastAPI）：检索 /search、问答 /ask、健康检查 /health。

启动：uv run ragbase serve [--host 127.0.0.1] [--port 8000]（默认读取 RAG_HOST / RAG_PORT；也可 uv run uvicorn ragbase.api.app:app）

必须单进程（不要加 --workers）：embedding / reranker（本地模型时）/ BM25 索引都加载在本进程内，
多 worker 会让每个进程各占一份模型显存，且 MPS 模型本身不能并发推理——
并发请求由线程池接住，召回 + 精排在 pipeline.lock 内串行，大模型生成在锁外并行。

对话 Agent（POST /chat、POST /chat/stream、GET /chat/sessions、GET|DELETE /chat/sessions/{id}）见 ragbase/agent/service.py：模型按需调用知识库检索工具，
会话（消息历史、会话摘要、引用编号）按会话保存在内置 SQLite 里（启动时打开，见 ragbase/db/），重启不丢，不自动淘汰；检索仍走 pipeline.lock，同会话的请求串行。
对话 Agent 未就绪（大模型未配置 / 索引为空）时，会话列表、查看历史和删除仍然可用（直接读库），只有发消息返回 503。

文档管理（POST /documents、POST /ingest、DELETE /documents/{name}）也在本进程内完成：
入库任务在后台线程里跑（同一时刻只有一个），每个批次单独持有 pipeline.lock，批间释放，/search 可以插进来；
任务结束后重建 BM25 快照并原子替换。命令行 ragbase ingest --full（会清空整个向量库）不能和服务同时运行；本地 Qdrant 目录同一时刻也只允许一个进程打开。

评测题集与体检（Web 体检页）：题集在内置 SQLite 的 eval_questions 表里，GET|POST /eval-questions、PUT|DELETE /eval-questions/{id}、
POST /eval-questions/import、GET /eval-questions/export；POST /diagnose 在后台线程里复用已加载的流水线做 RAG 体检（不调用大模型），
GET /diagnose/jobs/{job_id} 查进度，完成后报告存进数据库：GET /diagnose/reports、GET|DELETE /diagnose/reports/{id}。
体检与入库、删除文档共用一把“忙”锁，同一时刻只能有一个，冲突回 409；体检用开始时的题目快照，运行中改题集不影响本次体检。
题集和报告接口不依赖对话 Agent / 流水线，大模型未配置或索引为空时照常可用。

接口鉴权：设置 RAG_API_KEY 后，除前端页面 / 静态资源、/docs、/openapi.json 和精简版 /health 外，所有接口都要带
Authorization: Bearer <key>（或 X-API-Key）；默认拒绝 + 显式公开白名单，见 ragbase/api/auth.py。RAG_CORS_ORIGINS 配置跨域来源。

Web 界面：web/ 目录是前端工程，pnpm build 后产物在 web/dist（RAG_WEB_DIST 可改），本服务在同一端口托管它：
/assets/* 是静态资源，/、/library、/diagnose、/c/{session_id} 返回 index.html（前端路由）；API 路由不受影响。
"""
import inspect
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Path as PathParam, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ragbase.api.auth import (
    NO_AUTH_EXPOSED_WARNING,
    UNAUTHORIZED_DETAIL,
    is_authenticated,
    is_health,
    is_loopback_host,
    is_public,
)
from ragbase.api.schemas import (
    SESSION_ID_PATTERN,
    AskRequest,
    AskResponse,
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    DiagnoseJobResponse,
    DiagnoseRequest,
    DocumentInfo,
    DocumentsResponse,
    ErrorResponse,
    HealthResponse,
    LivenessResponse,
    JobResponse,
    ModelInfo,
    QuestionCreate,
    QuestionDeleteResponse,
    QuestionImportRequest,
    QuestionImportResponse,
    QuestionListResponse,
    QuestionResponse,
    QuestionUpdate,
    QuestionValidationResponse,
    ReportDeleteResponse,
    ReportListResponse,
    ReportResponse,
    SearchRequest,
    SearchResponse,
    SessionDeleteResponse,
    SessionListResponse,
    SessionResponse,
    UploadResponse,
    to_chunk,
    to_citation,
)
from ragbase.agent.service import ChatService, build_chat_service, delete_session_offline, read_session_history
from ragbase.agent.store import ChatStore
from ragbase.config import Settings, get_settings
from ragbase.db import Database, open_database
from ragbase.diagnostics import dedupe_questions, run_diagnosis
from ragbase.diagnostics_jobs import DiagnoseJob, DiagnoseJobManager, RequestCounter
from ragbase.diagnostics_reports import ReportStore
from ragbase.evaluation.questions import QuestionStore, QuestionValidationError
from ragbase.index_info import IndexRecorder
from ragbase.ingest.parser import LEGACY_EXTS, SUPPORTED_EXTS, sanitize_filename
from ragbase.ingest.jobs import BusyLock, IngestJobManager, JobRunningError, progress_handler
from ragbase.ingest.service import delete_document, run_ingest
from ragbase.logging_setup import request_id_var, setup_logging
from ragbase.ingest.common import collection_count
from ragbase.retrieval.pipeline import GenerationError, RAGPipeline, ask, build_pipeline, retrieve_and_rerank, should_refuse

logger = logging.getLogger(__name__)

INDEX_MISSING_MESSAGE = "索引不存在或为空，请先运行 ragbase ingest --full"
LLM_MISSING_MESSAGE = (
    "大模型未配置：请在 .env 中设置 RAG_LLM_API_KEY、RAG_LLM_MODEL 和 RAG_LLM_BASE_URL（旧的 ARK_* 变量同样有效）"
)
WEB_MISSING_HINT = "前端未构建：cd web && pnpm install && pnpm build"
LLM_FAILED_MESSAGE = "大模型调用失败，请稍后重试"
LLM_TIMEOUT_MESSAGE = "大模型响应超时，请稍后重试"
INTERNAL_ERROR_MESSAGE = "服务内部错误"
INGEST_UNAVAILABLE_MESSAGE = "索引不存在或为空，服务无法在线入库：请先运行 ragbase ingest 建立索引，再重启服务"
UPLOAD_CHUNK_BYTES = 1024 * 1024  # 上传流式写盘的块大小
UPLOAD_SLACK_BYTES = 64 * 1024  # Content-Length 比文件上限多出的 multipart 边界 / 字段开销

ERROR_CODES = {
    400: "bad_request",
    401: "unauthorized",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_error",
    502: "llm_error",
    504: "llm_timeout",
    503: "service_unavailable",
}
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")  # 只接受安全字符，避免日志注入


def default_pipeline_factory(settings: Settings) -> RAGPipeline:
    """真实流水线：加载本地模型；BM25 索引始终构建，以便按请求切换召回模式。"""
    pipeline, _ = build_pipeline(settings=settings, build_bm25=True)
    return pipeline


def default_chat_factory(pipeline: RAGPipeline, settings: Settings, db: Database | None = None) -> ChatService | None:
    """真实对话 Agent：会话持久化到 db；大模型未配置时为 None（/chat 返回 503）。"""
    return build_chat_service(pipeline, settings, db=db)


def _make_chat_service(chat_factory, pipeline: RAGPipeline, settings: Settings, db: Database) -> ChatService | None:
    """调用 chat_factory：能接收 db 参数（命名为 db，或带 **kwargs）的工厂收到 db=…，旧的 (pipeline, settings) 两参数工厂照旧调用。"""
    try:
        parameters = inspect.signature(chat_factory).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "db" in parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return chat_factory(pipeline, settings, db=db)
    return chat_factory(pipeline, settings)


def default_ingest_fn(pipeline: RAGPipeline, settings: Settings, progress):
    """真实入库：向量化用流水线里同一个 embedding（本地模型或云端 API），每个批次持 pipeline.lock；同时登记索引的 embedding 签名。"""
    store = pipeline.retriever.store
    recorder = IndexRecorder(settings, store)
    return run_ingest(
        settings.data_dir,
        store,
        recorder.wrap(pipeline.embeddings.embed_documents),
        settings.meta_file,
        lock=pipeline.lock,
        progress=progress,
        before_write=recorder.begin,
    )


def _error_response(status_code: int, detail: str, headers: dict | None = None) -> JSONResponse:
    error = ERROR_CODES.get(status_code, "internal_error" if status_code >= 500 else "http_error")
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail}, headers=headers)


def _validation_detail(exc: RequestValidationError) -> str:
    """只拼字段路径和原因，不回显用户输入。"""
    parts = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"] if item != "body")
        parts.append(f"{location}: {error['msg']}" if location else error["msg"])
    return "; ".join(parts)


def _device_text(settings: Settings) -> str:
    """流水线未就绪时 /health 显示的设备：全云端没有本地设备。"""
    return settings.device if settings.uses_local_models else "cloud"


def _bm25_ready(pipeline: RAGPipeline) -> bool:
    return bool(getattr(pipeline.retriever, "bm25_ready", False))


def _get_pipeline(request: Request) -> RAGPipeline:
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail=INDEX_MISSING_MESSAGE)
    return pipeline


def _check_mode(pipeline: RAGPipeline, mode: str | None, settings: Settings) -> None:
    """请求用到 BM25 的模式时，确认索引已构建。"""
    effective = mode or settings.mode
    if effective != "vector" and not _bm25_ready(pipeline):
        raise HTTPException(status_code=400, detail=f"服务未构建 BM25 索引，不支持 mode={effective}")


def _generation_http_error(exc: GenerationError) -> HTTPException:
    """大模型失败 -> 与 /ask 相同的状态码：超时 504，其余 502（完整异常只进日志，响应里只给固定提示）。"""
    if exc.timeout:
        return HTTPException(status_code=504, detail=LLM_TIMEOUT_MESSAGE)
    return HTTPException(status_code=502, detail=LLM_FAILED_MESSAGE)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _score_text(score: float | None) -> str:
    return "-" if score is None else f"{score:.4f}"


def _check_sources(pipeline: RAGPipeline, sources: list[str] | None) -> list[str] | None:
    """校验 sources 里的书名都在索引里；空 / 缺省表示不过滤。未知书名返回 400 并列出可选项。"""
    if not sources:
        return None
    names = list(dict.fromkeys(sources))
    valid = [document["name"] for document in pipeline.retriever.catalog()]
    unknown = [name for name in names if name not in valid]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"未知的书名 {unknown}；可选：{valid}（见 GET /documents）",
        )
    return names


def _sources_log(sources: list[str] | None) -> str:
    return f" sources={len(sources)}" if sources else ""


def _iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")


def _save_upload(upload: UploadFile, directory: Path, max_bytes: int, limit_mb: int) -> Path:
    """把上传内容流式写到数据目录下的隐藏临时文件（不整体读进内存），超限 / 为空则删除并报错。"""
    temporary = directory / f".upload-{uuid.uuid4().hex}.tmp"  # 以 . 开头：list_documents 不会把它当文档
    size = 0
    try:
        with open(temporary, "wb") as out:
            while chunk := upload.file.read(UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status_code=413, detail=f"文件超过大小上限 {limit_mb} MB")
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="上传的文件为空")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def create_app(pipeline_factory=None, settings: Settings | None = None, ingest_fn=None, chat_factory=None) -> FastAPI:
    """应用工厂。pipeline_factory(settings) -> RAGPipeline、ingest_fn(pipeline, settings, progress) -> IngestResult、
    chat_factory(pipeline, settings[, db=Database]) -> ChatService | None（声明了 db 参数的工厂会收到应用打开的数据库，用它持久化会话），
    测试里可注入假实现避免加载模型 / 真实入库 / 调用大模型。"""
    settings = settings or get_settings()
    pipeline_factory = pipeline_factory or default_pipeline_factory
    ingest_fn = ingest_fn or default_ingest_fn
    chat_factory = chat_factory or default_chat_factory
    setup_logging(settings.log_level)
    max_upload_bytes = settings.max_upload_mb * 1024 * 1024
    api_key = settings.api_key  # None = 鉴权关闭
    busy = BusyLock()  # 入库、删除文档、体检共用：同一时刻只能有一个
    jobs = IngestJobManager(busy)
    diagnose_jobs = DiagnoseJobManager(busy)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = None
        app.state.chat = None
        app.state.db = db = open_database(settings)  # 目录不可写 / 文件损坏时抛 DatabaseError（中文说明），服务不启动
        app.state.chat_store = ChatStore(db)
        app.state.question_store = QuestionStore(db)
        app.state.report_store = ReportStore(db)
        try:
            count = collection_count(settings)
            if settings.data_dir.is_dir():
                for leftover in settings.data_dir.glob(".upload-*.tmp"):  # 上次上传中途崩溃留下的临时文件
                    leftover.unlink(missing_ok=True)
            if not count:
                logger.warning("%s；服务仍会启动，/search 与 /ask 返回 503", INDEX_MISSING_MESSAGE)
            else:
                start = time.perf_counter()
                app.state.pipeline = pipeline_factory(settings)
                app.state.model_load_seconds = time.perf_counter() - start  # 体检报告里的“模型加载耗时”
                logger.info("流水线就绪：%d 个片段，启动加载用时 %.1fs", count, app.state.model_load_seconds)
                app.state.chat = _make_chat_service(chat_factory, app.state.pipeline, settings, db)
                if app.state.chat is None:
                    logger.warning("大模型未配置，对话 Agent 未启用（/chat 返回 503）")
            yield
            store = getattr(getattr(app.state.pipeline, "retriever", None), "store", None)
            if store is not None:
                try:
                    store.close()  # 释放本地向量库的目录锁
                except Exception:
                    logger.exception("关闭向量库失败")
        finally:
            try:
                if app.state.chat is not None:
                    app.state.chat.close()  # 先关 checkpointer 的独立连接，再关主连接
            except Exception:
                logger.exception("关闭对话服务失败")
            db.close()

    app = FastAPI(title="RAG 服务", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.pipeline = None
    app.state.chat = None
    app.state.db = None
    app.state.chat_store = None
    app.state.question_store = None
    app.state.report_store = None
    app.state.jobs = jobs
    app.state.diagnose_jobs = diagnose_jobs
    app.state.chat_requests = RequestCounter()  # /chat 与 /chat/stream 的累计请求数，体检用它判断运行期间有没有对话在抢资源
    app.state.model_load_seconds = 0.0

    def _reject_oversized_upload(request: Request):
        """POST /documents 的 Content-Length 明显超限时，不等整个请求体传完就拒绝（精确大小在写盘时再校验）。"""
        if request.method == "POST" and request.url.path == "/documents":
            length = request.headers.get("content-length", "")
            if length.isdigit() and int(length) > max_upload_bytes + UPLOAD_SLACK_BYTES:
                return _error_response(413, f"文件超过大小上限 {settings.max_upload_mb} MB")
        return None

    def _check_auth(request: Request):
        """接口鉴权（默认拒绝）：公开白名单之外的请求必须带有效密钥，否则 401；放在读请求体之前，所以未鉴权的上传 / 流式请求不会被处理。
        /health 总是放行，但把 request.state.authenticated 交给路由决定返回精简还是完整内容。不记录请求头里的任何值。"""
        request.state.authenticated = is_authenticated(request.headers, api_key)
        if request.state.authenticated or is_public(request.method, request.url.path) or is_health(
            request.method, request.url.path
        ):
            return None
        return _error_response(401, UNAUTHORIZED_DETAIL, headers={"WWW-Authenticate": "Bearer"})

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """请求 ID（沿用或生成 X-Request-ID 并回写）+ 接口鉴权 + 每个请求一行访问日志 + 兜底未处理异常。"""
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if _VALID_REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        request.state.log_extra = ""
        start = time.perf_counter()
        try:
            try:
                response = _check_auth(request) or _reject_oversized_upload(request) or await call_next(request)
            except Exception:
                logger.exception("未处理的异常：%s %s", request.method, request.url.path)
                response = _error_response(500, INTERNAL_ERROR_MESSAGE)
            response.headers["X-Request-ID"] = request_id
            logger.info(
                "%s %s -> %d %.1fms%s",
                request.method,
                request.url.path,
                response.status_code,
                (time.perf_counter() - start) * 1000,
                request.state.log_extra,
            )
            return response
        finally:
            request_id_var.reset(token)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return _error_response(exc.status_code, str(exc.detail), exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return _error_response(422, _validation_detail(exc))

    @app.exception_handler(QuestionValidationError)
    async def question_validation_handler(request: Request, exc: QuestionValidationError):
        """题集校验失败 -> 422：统一错误格式 {error, detail} 之外多一个逐题错误列表 errors（与上面的 RequestValidationError 处理器互不影响）。"""
        errors = exc.errors
        shown = "；".join(str(QuestionValidationError([error])) for error in errors[:3])
        detail = f"题集校验未通过（共 {len(errors)} 处）：{shown}" + ("……" if len(errors) > 3 else "")
        return JSONResponse(status_code=422, content={"error": "validation_error", "detail": detail, "errors": errors})

    error_models = {code: {"model": ErrorResponse} for code in (400, 401, 404, 409, 413, 422, 502, 503, 504)}
    question_error_models = {**error_models, 422: {"model": QuestionValidationResponse}}

    if settings.cors_origins:  # 最后添加 = 最外层：预检请求不经过鉴权，401 等响应也带 CORS 头
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["Authorization", "X-API-Key", "Content-Type"],
            expose_headers=["X-Request-ID"],
            allow_credentials=False,  # 密钥走请求头，不依赖 cookie
        )

    @app.get("/health", response_model=HealthResponse | LivenessResponse)
    def health(request: Request):
        """服务状态：索引 / 模型都就绪才是 ok；索引缺失时服务仍可启动，这里报告 not_ready。
        启用鉴权后，没有有效密钥只返回 {status, auth_required}（存活探针用），有效密钥才返回完整内容。"""
        pipeline = request.app.state.pipeline
        if not request.state.authenticated:
            return LivenessResponse(status="ok" if pipeline is not None else "not_ready", auth_required=True)
        count = collection_count(settings)
        retriever = pipeline.retriever if pipeline is not None else None
        return HealthResponse(
            status="ok" if pipeline is not None and count else "not_ready",
            chunk_count=count,
            bm25_ready=pipeline is not None and _bm25_ready(pipeline),
            bm25_built_at=_iso(getattr(retriever, "bm25_built_at", None)),
            bm25_chunk_count=getattr(retriever, "bm25_chunk_count", None),
            device=(pipeline.device if pipeline is not None and pipeline.device else _device_text(settings)),
            embedding=ModelInfo(provider=settings.embedding_provider, model=settings.embedding_model),
            reranker=ModelInfo(
                provider=settings.reranker_provider,
                model=None if settings.reranker_provider == "none" else settings.reranker_model,
            ),
            mode=settings.mode,
            llm_configured=(
                pipeline.answer_chain is not None if pipeline is not None else settings.llm_configured
            ),
            vector_store=settings.vector_store,
            llm_protocol=settings.llm_protocol,
            llm_model=settings.llm_model,  # 模型名不是密钥
            agent_ready=request.app.state.chat is not None,
            auth_required=api_key is not None,
        )

    @app.post(
        "/search",
        response_model=SearchResponse,
        response_model_exclude_unset=True,  # 未请求 candidates 时整个字段不出现
        responses=error_models,
    )
    def search(body: SearchRequest, request: Request):
        """召回 + 精排，不调用大模型。"""
        pipeline = _get_pipeline(request)
        _check_mode(pipeline, body.mode, settings)
        sources = _check_sources(pipeline, body.sources)
        min_score = settings.min_score if body.min_score is None else body.min_score
        start = time.perf_counter()
        reranked, candidates = retrieve_and_rerank(
            pipeline, body.query, body.top_k or settings.rerank_k, body.mode, sources
        )
        refused = should_refuse(reranked, min_score)
        top_score = reranked[0].metadata["rerank_score"] if reranked else None
        request.state.log_extra = f" refused={refused} top_score={_score_text(top_score)}" + _sources_log(sources)
        extra = {}
        if body.include_candidates:
            extra["candidates"] = [to_chunk(rank, doc) for rank, doc in enumerate(candidates, 1)]
        return SearchResponse(
            refused=refused,
            top_score=top_score,
            chunks=[to_chunk(rank, doc) for rank, doc in enumerate(reranked, 1)],
            took_ms=round((time.perf_counter() - start) * 1000, 1),
            **extra,
        )

    @app.post("/ask", response_model=AskResponse, responses=error_models)
    def ask_question(body: AskRequest, request: Request):
        """召回 + 精排 + 大模型生成；得分过低时拒答且不调用大模型。"""
        pipeline = _get_pipeline(request)
        if pipeline.answer_chain is None:
            raise HTTPException(status_code=503, detail=LLM_MISSING_MESSAGE)
        _check_mode(pipeline, body.mode, settings)
        sources = _check_sources(pipeline, body.sources)
        start = time.perf_counter()
        try:
            result = ask(
                pipeline,
                body.query,
                rerank_k=settings.rerank_k,
                min_score=settings.min_score if body.min_score is None else body.min_score,
                context_min_score=(
                    settings.context_min_score if body.context_min_score is None else body.context_min_score
                ),
                mode=body.mode,
                sources=sources,
            )
        except GenerationError as exc:
            logger.exception("大模型调用失败")  # 完整异常只进日志，响应里只给固定提示
            if exc.timeout:  # 超时（已按 RAG_LLM_MAX_RETRIES 重试过）单独用 504，便于客户端区分
                raise HTTPException(status_code=504, detail=LLM_TIMEOUT_MESSAGE) from exc
            raise HTTPException(status_code=502, detail=LLM_FAILED_MESSAGE) from exc
        request.state.log_extra = (
            f" refused={result.refused} top_score={_score_text(result.top_score)}" + _sources_log(sources)
        )
        return AskResponse(
            answer=result.answer,
            evidence=result.evidence,
            refused=result.refused,
            top_score=result.top_score,
            citations=[to_citation(n, doc) for n, doc in enumerate(result.docs, 1)],
            took_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    def _get_chat(request: Request) -> ChatService:
        _get_pipeline(request)  # 索引缺失 -> 503
        service = request.app.state.chat
        if service is None:
            raise HTTPException(status_code=503, detail=LLM_MISSING_MESSAGE)
        return service

    @app.post("/chat", response_model=ChatResponse, responses=error_models)
    def chat(body: ChatRequest, request: Request):
        """多轮对话：Agent 按需调用知识库工具（检索 / 列书目）后回答；session_id 相同的请求共享历史，同会话的请求串行处理。"""
        request.app.state.chat_requests.increment()
        service = _get_chat(request)
        sources = _check_sources(request.app.state.pipeline, body.sources)
        session_id = body.session_id or uuid.uuid4().hex
        start = time.perf_counter()
        try:
            result = service.chat(session_id, body.message, sources)
        except GenerationError as exc:
            logger.exception("对话模型调用失败")
            raise _generation_http_error(exc) from exc
        request.state.log_extra = f" session={session_id} tools={len(result.tool_calls)}" + _sources_log(sources)
        return ChatResponse(
            session_id=session_id,
            answer=result.answer,
            citations=[to_citation(n, doc) for n, doc in result.citations],
            tool_calls=result.tool_calls,
            took_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    @app.post("/chat/stream", responses={**error_models, 200: {"content": {"text/event-stream": {}}}})
    def chat_stream(body: ChatRequest, request: Request):
        """同 /chat，但以 Server-Sent Events 流式返回。事件：session、tool_call、token、citations、done、error。

        校验类错误（400 / 422 / 503）在开始流式之前以普通 JSON 返回；流式过程中的错误以 error 事件发出（data 同统一错误 JSON）。
        """
        request.app.state.chat_requests.increment()
        service = _get_chat(request)
        sources = _check_sources(request.app.state.pipeline, body.sources)
        session_id = body.session_id or uuid.uuid4().hex
        request.state.log_extra = f" session={session_id}" + _sources_log(sources)

        def events():  # 同步生成器：Starlette 在线程池里迭代它，所以阻塞的 agent 流不会卡住事件循环
            start = time.perf_counter()
            try:
                for event, data in service.stream(session_id, body.message, sources):
                    if event == "citations":
                        data = {"citations": [to_citation(n, doc).model_dump() for n, doc in data["citations"]]}
                    elif event == "done":
                        data = {**data, "took_ms": round((time.perf_counter() - start) * 1000, 1)}
                    yield _sse(event, data)
            except GenerationError as exc:
                logger.exception("对话模型调用失败")
                error = _generation_http_error(exc)
                yield _sse("error", {"error": ERROR_CODES[error.status_code], "detail": error.detail})
            except Exception:
                logger.exception("对话流处理失败")
                yield _sse("error", {"error": "internal_error", "detail": INTERNAL_ERROR_MESSAGE})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/chat/sessions", response_model=SessionListResponse, responses=error_models)
    def list_chat_sessions(request: Request):
        """会话列表（按最近使用倒序，供前端侧边栏）。直接读库，对话 Agent 未就绪（大模型未配置 / 索引缺失）时同样可用。"""
        service = request.app.state.chat
        sessions = service.list_sessions() if service is not None else request.app.state.chat_store.list_sessions()
        return SessionListResponse(sessions=[{**item, "updated_at": _iso(item["updated_at"])} for item in sessions])

    @app.get("/chat/sessions/{session_id}", response_model=SessionResponse, responses=error_models)
    def get_chat_session(request: Request, session_id: str = PathParam(pattern=SESSION_ID_PATTERN)):
        """会话的全部消息历史（用户 / 助手 / 工具；工具消息只给摘要）。会话不存在或已删除返回 404。对话 Agent 未就绪时直接读库。"""
        service = request.app.state.chat
        if service is not None:
            messages = service.history(session_id)
        else:
            messages = read_session_history(request.app.state.db, request.app.state.chat_store, session_id)
        if messages is None:
            raise HTTPException(status_code=404, detail=f"没有这个会话：{session_id}")
        messages = [
            {**m, "citations": [to_citation(n, doc) for n, doc in m["citations"]]} if "citations" in m else m
            for m in messages
        ]
        return SessionResponse(session_id=session_id, messages=messages)

    @app.delete("/chat/sessions/{session_id}", response_model=SessionDeleteResponse, responses=error_models)
    def delete_chat_session(request: Request, session_id: str = PathParam(pattern=SESSION_ID_PATTERN)):
        """删除会话（消息历史、引用、会话记录一并删除；等该会话正在处理的请求结束后执行）。对话 Agent 未就绪时同样可用。"""
        service = request.app.state.chat
        if service is not None:
            deleted = service.delete(session_id)
        else:
            deleted = delete_session_offline(request.app.state.db, request.app.state.chat_store, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"没有这个会话：{session_id}")
        return SessionDeleteResponse(session_id=session_id, deleted=True)

    def _ingest_runner(pipeline: RAGPipeline):
        """后台任务体：入库 -> 有变化则重建 BM25 快照（原子替换）-> 有文件失败则整体标记 failed。"""

        def run(job):
            handler = progress_handler(job)

            def on_progress(event, **info):
                handler(event, **info)
                if event == "file_failed":
                    job.update(skipped_files=[*job.skipped_files, f"{info['name']}（入库失败）"])

            result, failure = None, None
            try:
                result = ingest_fn(pipeline, settings, on_progress)
            except Exception as exc:
                logger.exception("入库中断")
                failure = exc
            if result is None or result.changed:  # 异常中断时索引也可能已部分改动，同样刷新
                job.update(current_file="重建 BM25 索引")
                try:
                    pipeline.retriever.refresh()
                except Exception as exc:
                    logger.exception("BM25 索引刷新失败")
                    failure = failure or RuntimeError(f"入库已完成，但 BM25 索引刷新失败：{exc}")
            if failure is not None:
                raise failure
            if result.failed_files:
                details = "; ".join(f"{name}: {reason}" for name, reason in result.failed_files.items())
                raise RuntimeError(f"{len(result.failed_files)} 个文件入库失败（其余文件已入库）：{details}")

        return run

    def _start_job(pipeline: RAGPipeline, conflict_detail=None):
        """conflict_detail(exc) -> 409 的说明；缺省用 JobRunningError 自带的（区分“入库任务”和“体检”）。"""
        try:
            return jobs.start(_ingest_runner(pipeline))
        except JobRunningError as exc:
            raise HTTPException(status_code=409, detail=conflict_detail(exc) if conflict_detail else str(exc)) from exc

    def _get_ingest_pipeline(request: Request) -> RAGPipeline:
        if request.app.state.pipeline is None:
            raise HTTPException(status_code=503, detail=INGEST_UNAVAILABLE_MESSAGE)
        return request.app.state.pipeline

    @app.post("/ingest", status_code=202, response_model=JobResponse, responses=error_models)
    def start_ingest(request: Request):
        """对数据目录做一次增量入库（新增 / 修改 / 删除的文件）；后台运行，用 GET /ingest/jobs/{job_id} 查进度。"""
        pipeline = _get_ingest_pipeline(request)
        return JobResponse(**_start_job(pipeline).to_dict())

    @app.get("/ingest/jobs/{job_id}", response_model=JobResponse, responses=error_models)
    def get_ingest_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"没有这个入库任务：{job_id}")
        return JobResponse(**job.to_dict())

    @app.get("/documents", response_model=DocumentsResponse, responses=error_models)
    def list_indexed_documents(request: Request):
        """已入库的文档（按文件名汇总）。数据来自检索器内存中的语料，旧版索引（无 fid / 章节）同样可用。"""
        pipeline = _get_pipeline(request)
        return DocumentsResponse(documents=[DocumentInfo(**entry) for entry in pipeline.retriever.catalog()])

    @app.post("/documents", status_code=202, response_model=UploadResponse, responses=error_models)
    def upload_document(request: Request, file: UploadFile = File(...)):
        """上传一个文档到数据目录并立即开始增量入库（同名文件覆盖）。已有任务在运行时文件仍会保存，返回 409。"""
        pipeline = _get_ingest_pipeline(request)
        name = sanitize_filename(file.filename)
        if name is None:
            raise HTTPException(status_code=400, detail="文件名无效：不能为空，不能以 . 或 ~$ 开头，不能是 README")
        extension = Path(name).suffix.lower()
        if extension in LEGACY_EXTS:
            raise HTTPException(status_code=400, detail=f"不支持旧版 {extension}：请先另存为 .docx 再上传")
        if extension not in SUPPORTED_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的格式 {extension or '（无扩展名）'}；支持：{'、'.join(sorted(SUPPORTED_EXTS))}",
            )
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        temporary = _save_upload(file, settings.data_dir, max_upload_bytes, settings.max_upload_mb)
        try:
            os.replace(temporary, settings.data_dir / name)  # 原子落盘；同名覆盖
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        job = _start_job(
            pipeline,
            conflict_detail=lambda exc: f"文件 {name} 已保存，但当前有{exc.label}在运行：请等它结束后调用 POST /ingest",
        )
        return UploadResponse(filename=name, **job.to_dict())

    @app.delete("/documents/{name}", response_model=DeleteResponse, responses=error_models)
    def delete_indexed_document(name: str, request: Request):
        """删除文档：源文件、索引中的片段（有 fid 按 fid，旧版片段按 source 文件名）和元数据，并刷新 BM25。"""
        pipeline = _get_ingest_pipeline(request)
        if sanitize_filename(name) != name:
            raise HTTPException(status_code=404, detail=f"没有这个文档：{name}")
        retriever = pipeline.retriever
        indexed = any(entry["name"] == name for entry in retriever.catalog())
        if not indexed and not (settings.data_dir / name).is_file():
            raise HTTPException(status_code=404, detail=f"没有这个文档：{name}")
        try:
            with jobs.exclusive():
                count = delete_document(
                    name,
                    settings.data_dir,
                    retriever.store,
                    settings.meta_file,
                    lock=pipeline.lock,
                    sources=retriever.resolve_sources([name]) if indexed else None,
                )
                retriever.refresh()  # 同步重建（分词缓存未命中时可能要几十秒）
        except JobRunningError as exc:
            raise HTTPException(status_code=409, detail=f"{exc}，请稍后再删除") from exc
        return DeleteResponse(name=name, chunks_deleted=count)

    # ------------------------------------------------------------------ 评测题集

    def _question_store(request: Request) -> QuestionStore:
        return request.app.state.question_store

    @app.get("/eval-questions", response_model=QuestionListResponse, responses=error_models)
    def list_eval_questions(request: Request):
        """评测题集（按 id 升序）。"""
        return QuestionListResponse(items=_question_store(request).list())

    @app.post("/eval-questions", status_code=201, response_model=QuestionResponse, responses=question_error_models)
    def create_eval_question(body: QuestionCreate, request: Request):
        """新增一道题，id 自动分配；字段校验失败 422（带 errors）。"""
        return _question_store(request).create(body.model_dump(exclude_unset=True))

    # 固定路径放在 /eval-questions/{question_id} 之前（两者方法也不同，双重保险）
    @app.post("/eval-questions/import", response_model=QuestionImportResponse, responses=question_error_models)
    def import_eval_questions(body: QuestionImportRequest, request: Request):
        """批量导入（JSON 数组，格式同导出文件）。校验与写入在同一事务里：有任何一题不合法就整体不写入，返回 422 与逐题错误。"""
        return _question_store(request).import_items(body.items, body.mode)

    @app.get("/eval-questions/export", response_model=list[dict], responses=error_models)
    def export_eval_questions(request: Request):
        """导出整个题集：JSON 数组，作为附件下载（eval_questions.json），旧版 ragbase eval 也能读。"""
        return JSONResponse(
            _question_store(request).export(),
            headers={"Content-Disposition": 'attachment; filename="eval_questions.json"'},
        )

    @app.put("/eval-questions/{question_id}", response_model=QuestionResponse, responses=question_error_models)
    def update_eval_question(question_id: int, body: QuestionUpdate, request: Request):
        """只改请求里出现的字段（传 null 清空可选字段）。"""
        updated = _question_store(request).update(question_id, body.model_dump(exclude_unset=True))
        if updated is None:
            raise HTTPException(status_code=404, detail=f"没有这道题：{question_id}")
        return updated

    @app.delete("/eval-questions/{question_id}", response_model=QuestionDeleteResponse, responses=error_models)
    def delete_eval_question(question_id: int, request: Request):
        if not _question_store(request).delete(question_id):
            raise HTTPException(status_code=404, detail=f"没有这道题：{question_id}")
        return QuestionDeleteResponse(deleted=True)

    # ------------------------------------------------------------------ 体检

    def _diagnose_runner(pipeline: RAGPipeline, questions: list[dict], duplicates: int, mode: str):
        """后台任务体：用开始时的题目快照跑体检，报告存进数据库，返回报告 id。"""
        db = app.state.db
        chat_requests = app.state.chat_requests

        def run(job: DiagnoseJob) -> int:
            report = run_diagnosis(
                pipeline,
                settings,
                questions,
                mode,
                app.state.model_load_seconds,
                duplicates,
                on_progress=lambda done, total: job.update(done=done, total=total),
                chat_requests=lambda: chat_requests.value,
            )
            return ReportStore(db).save(report, "web")

        return run

    @app.post("/diagnose", status_code=202, response_model=DiagnoseJobResponse, responses=error_models)
    def start_diagnose(request: Request, body: DiagnoseRequest | None = None):
        """开始一次体检（后台运行，不调用大模型）；用 GET /diagnose/jobs/{job_id} 查进度，完成后报告 id 在任务的 report_id 里。
        题目在这里取快照（先按 limit 截取、再去重），运行中改题集不影响本次体检。与入库 / 删除文档互斥，冲突 409。"""
        body = body or DiagnoseRequest()
        pipeline = _get_pipeline(request)  # 流水线未就绪 -> 503
        if not collection_count(settings):
            raise HTTPException(status_code=503, detail=INDEX_MISSING_MESSAGE)
        mode = body.mode or settings.mode
        _check_mode(pipeline, mode, settings)
        questions = _question_store(request).list()
        if body.limit is not None:
            questions = questions[: body.limit]
        questions, duplicates = dedupe_questions(questions)
        if not questions:
            raise HTTPException(status_code=400, detail="评测题集是空的：请先在体检页添加题目，或导入题集")
        try:
            job = diagnose_jobs.start(_diagnose_runner(pipeline, questions, duplicates, mode), total=len(questions))
        except JobRunningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return DiagnoseJobResponse(**job.to_dict())

    @app.get("/diagnose/jobs/{job_id}", response_model=DiagnoseJobResponse, responses=error_models)
    def get_diagnose_job(job_id: str):
        job = diagnose_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"没有这个体检任务：{job_id}")
        return DiagnoseJobResponse(**job.to_dict())

    @app.get("/diagnose/reports", response_model=ReportListResponse, responses=error_models)
    def list_diagnose_reports(request: Request):
        """体检报告列表（不含逐题记录，新的在前）。"""
        return ReportListResponse(items=request.app.state.report_store.list())

    @app.get("/diagnose/reports/{report_id}", response_model=ReportResponse, responses=error_models)
    def get_diagnose_report(report_id: int, request: Request):
        report = request.app.state.report_store.get(report_id)
        if report is None:
            raise HTTPException(status_code=404, detail=f"没有这份报告：{report_id}")
        return report

    @app.delete("/diagnose/reports/{report_id}", response_model=ReportDeleteResponse, responses=error_models)
    def delete_diagnose_report(report_id: int, request: Request):
        if not request.app.state.report_store.delete(report_id):
            raise HTTPException(status_code=404, detail=f"没有这份报告：{report_id}")
        return ReportDeleteResponse(deleted=True)

    _mount_web(app, settings.web_dist)
    return app


def _mount_web(app: FastAPI, dist: Path) -> None:
    """托管前端构建产物。放在所有 API 路由之后注册，且只匹配前端自己的路径（/assets/*、/、/library、/diagnose、/c/{id}），
    所以不会遮住 GET /documents、/health 这类 JSON 接口。dist 是否存在在请求时才检查：服务运行中重新 pnpm build 也立即生效。"""
    index = dist / "index.html"

    def page():
        if index.is_file():
            return FileResponse(index, headers={"Cache-Control": "no-cache"})  # 入口页不缓存，哈希资源可长缓存
        return HTMLResponse(f"<!doctype html><meta charset=utf-8><title>RAG 知识库</title><p>{WEB_MISSING_HINT}</p>")

    for route in ("/", "/library", "/diagnose", "/c/{session_id}"):  # GET /diagnose 是页面；同路径的 POST /diagnose 是 API
        app.add_api_route(route, page, methods=["GET"], include_in_schema=False)

    def asset(path: str):
        root = (dist / "assets").resolve()
        target = (root / path).resolve()
        if not target.is_relative_to(root) or not target.is_file():  # 目录穿越 / 不存在 -> 404（不回退 index.html）
            raise HTTPException(status_code=404, detail="资源不存在")
        return FileResponse(target, headers={"Cache-Control": "public, max-age=31536000, immutable"})  # 文件名带内容哈希

    app.add_api_route("/assets/{path:path}", asset, methods=["GET"], include_in_schema=False)


app = create_app()


def serve(host: str | None = None, port: int | None = None) -> None:
    """用 uvicorn 在当前进程里启动服务（ragbase serve）；host / port 缺省取配置（RAG_HOST / RAG_PORT）。"""
    runtime = get_settings()
    host = host or runtime.host
    if runtime.api_key is None and not is_loopback_host(host):
        logger.warning(NO_AUTH_EXPOSED_WARNING)
    uvicorn.run(
        app,
        host=host,
        port=port or runtime.port,
        log_level=runtime.log_level.lower(),
        access_log=False,  # 访问日志由上面的中间件统一输出（带请求 ID），避免重复
    )
