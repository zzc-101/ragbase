"""API 的单元测试：FastAPI TestClient + 注入的假流水线（不加载模型、不访问网络、不调用 LLM）。

运行：uv run python tests/test_api.py
"""
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from langchain_core.documents import Document

import api
from config import Settings, get_settings
from rag_pipeline import NO_ANSWER, RAGPipeline

ANSWER_RAW = "【依据】\n- [片段1] “原文一”\n【回答】\n基于依据的回答 [片段1]"


class _Retriever:
    """返回固定候选，并记录每次调用的 (question, mode)。"""

    def __init__(self, bm25_ready=True, delay=0.0):
        self.bm25_ready = bm25_ready
        self.delay = delay
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.fail = False
        self._guard = threading.Lock()

    def invoke(self, question, mode=None):
        self.calls.append((question, mode))
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.fail:
                raise RuntimeError("检索内部出错 sk-leak-123")
            time.sleep(self.delay)
            return [
                Document(
                    page_content=f"片段文本{i}",
                    metadata={
                        "chunk_id": f"c{i}",
                        "source": "/data/dir/book.pdf",
                        "page": 11 + i if i == 0 else None,  # 只有第 0 个片段带页码
                        "vec_rank": i + 1,
                        "bm25_rank": None,
                        "rrf_score": 0.03 - i * 0.01,
                    },
                )
                for i in range(3)
            ]
        finally:
            with self._guard:
                self.active -= 1


class _Reranker:
    def __init__(self, scores):
        self.scores = scores

    def rerank_with_scores(self, query, documents, top_k=5):
        return self.scores[:top_k]


class _Chain:
    def __init__(self, pipeline_ref=None, error=None):
        self.calls = []
        self.lock_held_during_call = None
        self.pipeline_ref = pipeline_ref
        self.error = error

    def invoke(self, payload):
        self.calls.append(payload)
        if self.pipeline_ref:
            self.lock_held_during_call = self.pipeline_ref[0].lock.locked()
        if self.error:
            raise self.error
        return ANSWER_RAW


def _pipeline(scores=((0, 0.99), (1, 0.96), (2, 0.5)), chain=True, retriever=None, error=None):
    holder = []
    pipeline = RAGPipeline(
        retriever=retriever or _Retriever(),
        reranker=_Reranker(list(scores)),
        answer_chain=_Chain(holder, error) if chain else None,
        embeddings=None,
        device="cpu",
    )
    holder.append(pipeline)
    return pipeline


def _settings(**overrides):
    values = dict(device="cpu", log_level="WARNING", chat_api_key="sk-test-FAKE-KEY")
    return Settings(_env_file=None, **{**values, **overrides})


@contextmanager
def _client(pipeline=None, count=6408, settings=None):
    """假索引计数 + 假流水线工厂，TestClient 以 with 方式启动以触发 lifespan。"""
    original = api.collection_count
    api.collection_count = lambda *args, **kwargs: count
    try:
        app = api.create_app(
            pipeline_factory=lambda _settings: pipeline,
            settings=settings or _settings(),
        )
        with TestClient(app) as client:
            yield client
    finally:
        api.collection_count = original


def _assert_error(response, status, error):
    assert response.status_code == status, (response.status_code, response.text)
    body = response.json()
    assert set(body) == {"error", "detail"}, body
    assert body["error"] == error and isinstance(body["detail"], str) and body["detail"], body
    return body


def test_health_ok():
    with _client(_pipeline()) as client:
        body = client.get("/health").json()
    assert body == {
        "status": "ok", "chunk_count": 6408, "bm25_ready": True,
        "device": "cpu", "mode": "hybrid", "llm_configured": True,
    }, body


def test_health_without_llm_and_bm25():
    with _client(_pipeline(chain=False, retriever=_Retriever(bm25_ready=False))) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok" and body["llm_configured"] is False and body["bm25_ready"] is False


def test_health_not_ready_when_index_missing():
    with _client(_pipeline(), count=None) as client:  # 索引不存在：不建流水线，但服务能启动
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "not_ready" and body["chunk_count"] is None, body
        assert body["bm25_ready"] is False
        for path in ("/search", "/ask"):
            response = client.post(path, json={"query": "什么是软件工程"})
            body = _assert_error(response, 503, "service_unavailable")
            assert body["detail"] == "索引不存在或为空，请先运行 uv run ingest.py"


def test_health_not_ready_when_index_empty():
    with _client(_pipeline(), count=0) as client:
        assert client.get("/health").json()["status"] == "not_ready"


def test_search_shape():
    pipeline = _pipeline()
    with _client(pipeline) as client:
        response = client.post("/search", json={"query": "  关键路径法怎么计算  "})
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"refused", "top_score", "chunks", "took_ms"}, body  # 默认不含 candidates
    assert body["refused"] is False and body["top_score"] == 0.99
    assert isinstance(body["took_ms"], float)
    assert len(body["chunks"]) == 3
    first, second = body["chunks"][0], body["chunks"][1]
    assert set(first) == {"rank", "chunk_id", "text", "source", "page", "scores"}
    assert first["rank"] == 1 and first["chunk_id"] == "c0" and first["text"] == "片段文本0"
    assert first["source"] == "book.pdf" and first["page"] == 12  # basename；page 为 1 起
    assert first["scores"] == {"vec_rank": 1, "bm25_rank": None, "rrf_score": 0.03, "rerank_score": 0.99}
    assert second["rank"] == 2 and second["page"] is None  # 元数据没有页码 -> null
    assert pipeline.retriever.calls == [("关键路径法怎么计算", None)]  # 问题已去空白，默认模式不覆盖


def test_search_top_k_and_include_candidates():
    with _client(_pipeline()) as client:
        body = client.post("/search", json={"query": "q", "top_k": 2, "include_candidates": True}).json()
    assert len(body["chunks"]) == 2
    candidates = body["candidates"]
    assert [c["rank"] for c in candidates] == [1, 2, 3]  # 精排前的 3 个候选，保持召回顺序
    assert all(c["scores"]["rerank_score"] is None for c in candidates)
    assert set(candidates[0]) == {"rank", "chunk_id", "text", "source", "page", "scores"}


def test_search_refused_flag_and_min_score_override():
    with _client(_pipeline(scores=[(1, 0.2), (0, 0.1)])) as client:
        refused = client.post("/search", json={"query": "红烧肉怎么做"}).json()
        relaxed = client.post("/search", json={"query": "红烧肉怎么做", "min_score": 0}).json()
        strict = client.post("/search", json={"query": "q", "min_score": 0.1}).json()
    assert refused["refused"] is True and refused["top_score"] == 0.2
    assert len(refused["chunks"]) == 2  # 拒答也返回片段，便于查看
    assert relaxed["refused"] is False and strict["refused"] is False


def test_search_validation_errors_use_unified_shape():
    bad_bodies = [
        {"query": ""},
        {"query": "   "},
        {"query": "长" * 1001},
        {},
        {"query": "q", "top_k": 0},
        {"query": "q", "mode": "nope"},
        {"query": "q", "min_score": 1.5},
    ]
    with _client(_pipeline()) as client:
        for payload in bad_bodies:
            body = _assert_error(client.post("/search", json=payload), 422, "validation_error")
            assert "长长长" not in body["detail"]  # 不回显用户输入
        assert "query" in client.post("/search", json={"query": " "}).json()["detail"]
        _assert_error(client.post("/ask", json={"query": ""}), 422, "validation_error")
        _assert_error(client.post("/search", content="不是JSON", headers={"content-type": "application/json"}), 422, "validation_error")


def test_search_mode_passed_through():
    pipeline = _pipeline()
    with _client(pipeline) as client:
        for mode in ("bm25", "vector", "hybrid"):
            assert client.post("/search", json={"query": "q", "mode": mode}).status_code == 200
    assert [mode for _, mode in pipeline.retriever.calls] == ["bm25", "vector", "hybrid"]


def test_mode_needing_bm25_rejected_when_index_not_built():
    pipeline = _pipeline(retriever=_Retriever(bm25_ready=False))
    with _client(pipeline) as client:
        body = _assert_error(client.post("/search", json={"query": "q", "mode": "bm25"}), 400, "bad_request")
        assert "BM25" in body["detail"]
        _assert_error(client.post("/search", json={"query": "q"}), 400, "bad_request")  # 默认 hybrid 也需要
        assert client.post("/search", json={"query": "q", "mode": "vector"}).status_code == 200
    assert [mode for _, mode in pipeline.retriever.calls] == ["vector"]


def test_ask_success_with_citations():
    pipeline = _pipeline()
    with _client(pipeline) as client:
        response = client.post("/ask", json={"query": "什么是软件工程", "mode": "vector"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"answer", "evidence", "refused", "top_score", "citations", "took_ms"}
    assert body["answer"] == "基于依据的回答 [片段1]" and body["evidence"] == "- [片段1] “原文一”"
    assert body["refused"] is False and body["top_score"] == 0.99
    # 默认 context_min_score=0.95：只有 0.99、0.96 两个片段交给 LLM，0.5 的被过滤
    assert [c["n"] for c in body["citations"]] == [1, 2]
    assert [c["chunk_id"] for c in body["citations"]] == ["c0", "c1"]
    assert body["citations"][0]["source"] == "book.pdf" and body["citations"][0]["page"] == 12
    assert body["citations"][0]["scores"]["rerank_score"] == 0.99
    context = pipeline.answer_chain.calls[0]["context"]
    assert "[片段1] 片段文本0" in context and "[片段2] 片段文本1" in context and "片段文本2" not in context
    assert pipeline.retriever.calls == [("什么是软件工程", "vector")]


def test_ask_context_min_score_override():
    pipeline = _pipeline()
    with _client(pipeline) as client:
        body = client.post("/ask", json={"query": "q", "context_min_score": 0}).json()
    assert [c["n"] for c in body["citations"]] == [1, 2, 3]


def test_ask_refused_does_not_call_llm():
    pipeline = _pipeline(scores=[(0, 0.05)])
    with _client(pipeline) as client:
        response = client.post("/ask", json={"query": "红烧肉怎么做"})
    body = response.json()
    assert response.status_code == 200
    assert body["refused"] is True and body["answer"] == NO_ANSWER and body["evidence"] == ""
    assert body["citations"] == [] and body["top_score"] == 0.05
    assert pipeline.answer_chain.calls == []


def test_ask_llm_not_configured_returns_503():
    pipeline = _pipeline(chain=False)
    with _client(pipeline) as client:
        body = _assert_error(client.post("/ask", json={"query": "q"}), 503, "service_unavailable")
        assert "大模型未配置" in body["detail"]
        assert client.post("/search", json={"query": "q"}).status_code == 200  # 检索不受影响
    assert pipeline.retriever.calls == [("q", None)]  # /ask 在检索前就拒绝了


def test_ask_llm_exception_returns_502_without_leaking():
    secret = "sk-ULTRA-SECRET-999"
    pipeline = _pipeline(error=RuntimeError(f"Incorrect API key provided: {secret}"))
    with _client(pipeline) as client:
        response = client.post("/ask", json={"query": "q"})
    body = _assert_error(response, 502, "llm_error")
    assert secret not in response.text and "RuntimeError" not in response.text, body
    assert len(pipeline.answer_chain.calls) == 1


def test_request_id_echoed_and_generated():
    with _client(_pipeline()) as client:
        echoed = client.get("/health", headers={"X-Request-ID": "req-abc_123"})
        generated = client.get("/health")
        unsafe = client.get("/health", headers={"X-Request-ID": "bad id\twith space"})
        failed = client.post("/search", json={"query": ""}, headers={"X-Request-ID": "err-1"})
    assert echoed.headers["X-Request-ID"] == "req-abc_123"
    assert len(generated.headers["X-Request-ID"]) == 32
    assert unsafe.headers["X-Request-ID"] != "bad id\twith space" and len(unsafe.headers["X-Request-ID"]) == 32
    assert failed.headers["X-Request-ID"] == "err-1"  # 错误响应也带


def test_unknown_route_and_method_use_unified_shape():
    with _client(_pipeline()) as client:
        _assert_error(client.get("/nope"), 404, "not_found")
        _assert_error(client.get("/search"), 405, "method_not_allowed")


def test_unhandled_error_returns_500_unified_and_logs_traceback():
    retriever = _Retriever()
    retriever.fail = True
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture(level=logging.INFO)
    logging.getLogger("api").addHandler(handler)
    try:
        with _client(_pipeline(retriever=retriever)) as client:
            response = client.post("/search", json={"query": "q"}, headers={"X-Request-ID": "boom-1"})
    finally:
        logging.getLogger("api").removeHandler(handler)
    body = _assert_error(response, 500, "internal_error")
    assert "sk-leak-123" not in response.text and "检索内部出错" not in response.text
    assert response.headers["X-Request-ID"] == "boom-1"
    traces = [r for r in records if r.exc_info]
    assert traces and "检索内部出错" in repr(traces[0].exc_info[1])  # 日志里有完整异常
    assert body["detail"] == "服务内部错误"


def test_access_log_line_has_method_path_status_latency_and_search_fields():
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture(level=logging.INFO)
    logger = logging.getLogger("api")
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        with _client(_pipeline(scores=[(0, 0.05)])) as client:
            client.post("/search", json={"query": "q"})
            client.get("/health")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    search_line = next(line for line in records if line.startswith("POST /search"))
    health_line = next(line for line in records if line.startswith("GET /health"))
    assert "-> 200" in search_line and "ms" in search_line, search_line
    assert "refused=True" in search_line and "top_score=0.0500" in search_line, search_line
    assert "refused" not in health_line and "-> 200" in health_line, health_line


def test_concurrent_retrieval_is_serialized_but_llm_runs_outside_lock():
    retriever = _Retriever(delay=0.05)
    pipeline = _pipeline(retriever=retriever)
    with _client(pipeline) as client:
        statuses = []

        def worker(path):
            statuses.append(client.post(path, json={"query": "q"}).status_code)

        threads = [threading.Thread(target=worker, args=("/search" if i % 2 else "/ask",)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert statuses == [200] * 6, statuses
    assert retriever.max_active == 1, retriever.max_active  # 召回 + 精排互斥
    assert len(retriever.calls) == 6
    assert pipeline.answer_chain.lock_held_during_call is False  # LLM 调用时没有持锁
    assert not pipeline.lock.locked()


def test_settings_repr_hides_secret():
    secret = "sk-test-VERY-SECRET-VALUE"
    saved = os.environ.get("ARK_CHAT_API_KEY")
    os.environ["ARK_CHAT_API_KEY"] = secret
    get_settings.cache_clear()
    try:
        settings = get_settings()
        real_key = settings.chat_api_key.get_secret_value()
        assert real_key == secret  # 环境变量优先于 .env
        for text in (repr(settings), str(settings), settings.model_dump_json(), str(settings.model_dump())):
            assert secret not in text, text
        assert "SecretStr" in repr(settings) or "**********" in repr(settings)
        assert settings.llm_configured in (True, False)
    finally:
        if saved is None:
            os.environ.pop("ARK_CHAT_API_KEY", None)
        else:
            os.environ["ARK_CHAT_API_KEY"] = saved
        get_settings.cache_clear()
    health_settings = _settings(chat_api_key=secret, chat_model="m", base_url="http://x")
    with _client(_pipeline(), settings=health_settings) as client:
        assert secret not in client.get("/health").text


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
