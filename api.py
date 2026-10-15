"""RAG HTTP 服务（FastAPI）：检索 /search、问答 /ask、健康检查 /health。

启动：uv run uvicorn api:app --host 127.0.0.1 --port 8000   （或 uv run api.py，读取 RAG_HOST / RAG_PORT）

必须单进程（不要加 --workers）：embedding / reranker / BM25 索引都加载在本进程内，
多 worker 会让每个进程各占一份模型显存，且 MPS 模型本身不能并发推理——
并发请求由线程池接住，召回 + 精排在 pipeline.lock 内串行，大模型生成在锁外并行。
"""
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api_schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    to_chunk,
    to_citation,
)
from config import Settings, get_settings
from logging_setup import request_id_var, setup_logging
from rag_common import collection_count
from rag_pipeline import GenerationError, RAGPipeline, ask, build_pipeline, retrieve_and_rerank, should_refuse

logger = logging.getLogger(__name__)

INDEX_MISSING_MESSAGE = "索引不存在或为空，请先运行 uv run ingest.py"
LLM_MISSING_MESSAGE = "大模型未配置：请在 .env 中设置 ARK_CHAT_API_KEY、ARK_CHAT_MODEL 和 ARK_BASE_URL"
LLM_FAILED_MESSAGE = "大模型调用失败，请稍后重试"
INTERNAL_ERROR_MESSAGE = "服务内部错误"

ERROR_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    422: "validation_error",
    502: "llm_error",
    503: "service_unavailable",
}
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")  # 只接受安全字符，避免日志注入


def default_pipeline_factory(settings: Settings) -> RAGPipeline:
    """真实流水线：加载本地模型；BM25 索引始终构建，以便按请求切换召回模式。"""
    pipeline, _ = build_pipeline(settings=settings, build_bm25=True)
    return pipeline


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


def create_app(pipeline_factory=None, settings: Settings | None = None) -> FastAPI:
    """应用工厂。pipeline_factory(settings) -> RAGPipeline，测试里可注入假流水线避免加载模型。"""
    settings = settings or get_settings()
    pipeline_factory = pipeline_factory or default_pipeline_factory
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = None
        count = collection_count()
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
                response = await call_next(request)
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

    error_models = {code: {"model": ErrorResponse} for code in (400, 422, 502, 503)}

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        """服务状态：索引 / 模型都就绪才是 ok；索引缺失时服务仍可启动，这里报告 not_ready。"""
        pipeline = request.app.state.pipeline
        count = collection_count()
        return HealthResponse(
            status="ok" if pipeline is not None and count else "not_ready",
            chunk_count=count,
            bm25_ready=pipeline is not None and _bm25_ready(pipeline),
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
        min_score = settings.min_score if body.min_score is None else body.min_score
        start = time.perf_counter()
        reranked, candidates = retrieve_and_rerank(
            pipeline, body.query, body.top_k or settings.rerank_k, body.mode
        )
        refused = should_refuse(reranked, min_score)
        top_score = reranked[0].metadata["rerank_score"] if reranked else None
        request.state.log_extra = f" refused={refused} top_score={_score_text(top_score)}"
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
            )
        except GenerationError as exc:
            logger.exception("大模型调用失败")  # 完整异常只进日志，响应里只给固定提示
            raise HTTPException(status_code=502, detail=LLM_FAILED_MESSAGE) from exc
        request.state.log_extra = f" refused={result.refused} top_score={_score_text(result.top_score)}"
        return AskResponse(
            answer=result.answer,
            evidence=result.evidence,
            refused=result.refused,
            top_score=result.top_score,
            citations=[to_citation(n, doc) for n, doc in enumerate(result.docs, 1)],
            took_ms=round((time.perf_counter() - start) * 1000, 1),
        )

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
