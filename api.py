"""RAG HTTP 服务（FastAPI）：检索 /search、问答 /ask、健康检查 /health。

启动：uv run uvicorn api:app --host 127.0.0.1 --port 8000   （或 uv run api.py，读取 RAG_HOST / RAG_PORT）

必须单进程（不要加 --workers）：embedding / reranker / BM25 索引都加载在本进程内，
多 worker 会让每个进程各占一份模型显存，且 MPS 模型本身不能并发推理——
并发请求由线程池接住，召回 + 精排在 pipeline.lock 内串行，大模型生成在锁外并行。

文档管理（POST /documents、POST /ingest、DELETE /documents/{name}）也在本进程内完成：
入库任务在后台线程里跑（同一时刻只有一个），每个批次单独持有 pipeline.lock，批间释放，/search 可以插进来；
任务结束后重建 BM25 快照并原子替换。命令行 ingest.py（会删除整个 chroma_db）不能和服务同时运行。
"""
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api_schemas import (
    AskRequest,
    AskResponse,
    DeleteResponse,
    DocumentInfo,
    DocumentsResponse,
    ErrorResponse,
    HealthResponse,
    JobResponse,
    SearchRequest,
    SearchResponse,
    UploadResponse,
    to_chunk,
    to_citation,
)
from config import Settings, get_settings
from document_parser import LEGACY_EXTS, SUPPORTED_EXTS, sanitize_filename
from ingest_jobs import IngestJobManager, JobRunningError, progress_handler
from ingest_service import delete_document, run_ingest
from logging_setup import request_id_var, setup_logging
from rag_common import collection_count
from rag_pipeline import GenerationError, RAGPipeline, ask, build_pipeline, retrieve_and_rerank, should_refuse

logger = logging.getLogger(__name__)

INDEX_MISSING_MESSAGE = "索引不存在或为空，请先运行 uv run ingest.py"
LLM_MISSING_MESSAGE = "大模型未配置：请在 .env 中设置 ARK_CHAT_API_KEY、ARK_CHAT_MODEL 和 ARK_BASE_URL"
LLM_FAILED_MESSAGE = "大模型调用失败，请稍后重试"
INTERNAL_ERROR_MESSAGE = "服务内部错误"
INGEST_UNAVAILABLE_MESSAGE = "索引不存在或为空，服务无法在线入库：请先运行 uv run ingest_incremental.py 建立索引，再重启服务"
UPLOAD_CHUNK_BYTES = 1024 * 1024  # 上传流式写盘的块大小
UPLOAD_SLACK_BYTES = 64 * 1024  # Content-Length 比文件上限多出的 multipart 边界 / 字段开销

ERROR_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_error",
    502: "llm_error",
    503: "service_unavailable",
}
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")  # 只接受安全字符，避免日志注入


def default_pipeline_factory(settings: Settings) -> RAGPipeline:
    """真实流水线：加载本地模型；BM25 索引始终构建，以便按请求切换召回模式。"""
    pipeline, _ = build_pipeline(settings=settings, build_bm25=True)
    return pipeline


def default_ingest_fn(pipeline: RAGPipeline, settings: Settings, progress):
    """真实入库：向量化用流水线里同一个 embedding 模型，每个批次持 pipeline.lock。"""
    return run_ingest(
        settings.data_dir,
        pipeline.retriever.collection,
        pipeline.embeddings.embed_documents,
        settings.meta_file,
        lock=pipeline.lock,
        progress=progress,
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


def create_app(pipeline_factory=None, settings: Settings | None = None, ingest_fn=None) -> FastAPI:
    """应用工厂。pipeline_factory(settings) -> RAGPipeline、ingest_fn(pipeline, settings, progress) -> IngestResult，
    测试里可注入假实现避免加载模型 / 真实入库。"""
    settings = settings or get_settings()
    pipeline_factory = pipeline_factory or default_pipeline_factory
    ingest_fn = ingest_fn or default_ingest_fn
    setup_logging(settings.log_level)
    max_upload_bytes = settings.max_upload_mb * 1024 * 1024
    jobs = IngestJobManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = None
        count = collection_count(str(settings.db_dir))
        if settings.data_dir.is_dir():
            for leftover in settings.data_dir.glob(".upload-*.tmp"):  # 上次上传中途崩溃留下的临时文件
                leftover.unlink(missing_ok=True)
        if not count:
            logger.warning("%s；服务仍会启动，/search 与 /ask 返回 503", INDEX_MISSING_MESSAGE)
        else:
            start = time.perf_counter()
            app.state.pipeline = pipeline_factory(settings)
            logger.info("流水线就绪：%d 个片段，启动加载用时 %.1fs", count, time.perf_counter() - start)
        yield

    app = FastAPI(title="RAG 服务", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.pipeline = None
    app.state.jobs = jobs

    def _reject_oversized_upload(request: Request):
        """POST /documents 的 Content-Length 明显超限时，不等整个请求体传完就拒绝（精确大小在写盘时再校验）。"""
        if request.method == "POST" and request.url.path == "/documents":
            length = request.headers.get("content-length", "")
            if length.isdigit() and int(length) > max_upload_bytes + UPLOAD_SLACK_BYTES:
                return _error_response(413, f"文件超过大小上限 {settings.max_upload_mb} MB")
        return None

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """请求 ID（沿用或生成 X-Request-ID 并回写）+ 每个请求一行访问日志 + 兜底未处理异常。"""
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if _VALID_REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        request.state.log_extra = ""
        start = time.perf_counter()
        try:
            try:
                response = _reject_oversized_upload(request) or await call_next(request)
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

    error_models = {code: {"model": ErrorResponse} for code in (400, 404, 409, 413, 422, 502, 503)}

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        """服务状态：索引 / 模型都就绪才是 ok；索引缺失时服务仍可启动，这里报告 not_ready。"""
        pipeline = request.app.state.pipeline
        count = collection_count(str(settings.db_dir))
        retriever = pipeline.retriever if pipeline is not None else None
        return HealthResponse(
            status="ok" if pipeline is not None and count else "not_ready",
            chunk_count=count,
            bm25_ready=pipeline is not None and _bm25_ready(pipeline),
            bm25_built_at=_iso(getattr(retriever, "bm25_built_at", None)),
            bm25_chunk_count=getattr(retriever, "bm25_chunk_count", None),
            device=(pipeline.device if pipeline is not None and pipeline.device else settings.device),
            mode=settings.mode,
            llm_configured=(
                pipeline.answer_chain is not None if pipeline is not None else settings.llm_configured
            ),
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

    def _start_job(pipeline: RAGPipeline, conflict_detail: str | None = None):
        try:
            return jobs.start(_ingest_runner(pipeline))
        except JobRunningError as exc:
            raise HTTPException(status_code=409, detail=conflict_detail or str(exc)) from exc

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
            conflict_detail=f"文件 {name} 已保存，但当前有入库任务在运行：请等它结束后调用 POST /ingest",
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
                    retriever.collection,
                    settings.meta_file,
                    lock=pipeline.lock,
                    sources=retriever.resolve_sources([name]) if indexed else None,
                )
                retriever.refresh()  # 同步重建（分词缓存未命中时可能要几十秒）
        except JobRunningError as exc:
            raise HTTPException(status_code=409, detail=f"{exc}，请稍后再删除") from exc
        return DeleteResponse(name=name, chunks_deleted=count)

    return app


app = create_app()


if __name__ == "__main__":
    runtime = get_settings()
    uvicorn.run(
        app,
        host=runtime.host,
        port=runtime.port,
        log_level=runtime.log_level.lower(),
        access_log=False,  # 访问日志由上面的中间件统一输出（带请求 ID），避免重复
    )
