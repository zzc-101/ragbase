"""API 的单元测试：FastAPI TestClient + 注入的假流水线（不加载模型、不访问网络、不调用 LLM）。

运行：uv run python tests/test_api.py
"""
import logging
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from ragbase.api import app as api
from ragbase.config import Settings, get_settings
from ragbase.ingest.service import IngestResult
from ragbase.retrieval.pipeline import NO_ANSWER, RAGPipeline

# 所有测试默认把数据 / 索引 / 元数据目录指向这个临时目录，绝不碰真实的 data/、chroma_db/
_SCRATCH = tempfile.TemporaryDirectory(prefix="ragdemo-api-test-")
SCRATCH = Path(_SCRATCH.name)

ANSWER_RAW = "【依据】\n- [片段1] “原文一”\n【回答】\n基于依据的回答 [片段1]"


class _Retriever:
    """返回固定候选，并记录每次调用的 (question, mode)；sources 单独记在 source_calls。

    三个候选：c0（book.pdf，带页码和章节路径）、c1（book.pdf，旧版风格：无页码无章节）、c2（other.pdf）。
    """

    NAMES = ["book.pdf", "other.pdf"]

    def __init__(self, bm25_ready=True, delay=0.0):
        self.bm25_ready = bm25_ready
        self.delay = delay
        self.calls = []
        self.source_calls = []
        self.active = 0
        self.max_active = 0
        self.fail = False
        self._guard = threading.Lock()
        self.bm25_built_at = 1_760_000_000.0 if bm25_ready else None  # 2025-10-09T08:53:20Z
        self.bm25_chunk_count = 6408 if bm25_ready else None
        self.refreshes = 0
        self.fail_refresh = False
        self.store = None

    def catalog(self):
        return [
            {"name": name, "chunk_count": 10, "heading_chunks": 4 if name == "book.pdf" else 0,
             "has_headings": name == "book.pdf"}
            for name in self.NAMES
        ]

    def resolve_sources(self, names):
        return [f"/data/dir/{name}" for name in names]

    def refresh(self):
        if self.fail_refresh:
            raise RuntimeError("刷新失败")
        self.refreshes += 1
        self.bm25_built_at += 60

    def invoke(self, question, mode=None, sources=None):
        self.calls.append((question, mode))
        self.source_calls.append(sources)
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.fail:
                raise RuntimeError("检索内部出错 sk-leak-123")
            time.sleep(self.delay)
            docs = [
                Document(
                    page_content=f"片段文本{i}",
                    metadata={
                        "chunk_id": f"c{i}",
                        "source": "/data/dir/other.pdf" if i == 2 else "/data/dir/book.pdf",
                        "page": 11 + i if i == 0 else None,  # 只有第 0 个片段带页码
                        **({"heading_path": "第1章 概述 > 1.1 范围"} if i == 0 else {}),  # 只有第 0 个片段带章节
                        "vec_rank": i + 1,
                        "bm25_rank": None,
                        "rrf_score": 0.03 - i * 0.01,
                    },
                )
                for i in range(3)
            ]
            if sources:
                docs = [doc for doc in docs if Path(doc.metadata["source"]).name in sources]
            return docs
        finally:
            with self._guard:
                self.active -= 1


class _Reranker:
    def __init__(self, scores):
        self.scores = scores

    def rerank_with_scores(self, query, documents, top_k=5):
        return [(index, score) for index, score in self.scores if index < len(documents)][:top_k]  # 过滤后候选变少


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
    values = dict(
        device="cpu", log_level="WARNING", vector_store="chroma", llm_api_key="sk-test-FAKE-KEY", llm_model="test-model", llm_base_url=None,  # None：不受环境变量影响
        data_dir=SCRATCH / "default-data", db_dir=SCRATCH / "default-chroma",
        meta_file=SCRATCH / "default-meta.json", cache_dir=SCRATCH / "default-cache",
    )
    return Settings(_env_file=None, **{**values, **overrides})


@contextmanager
def _client(pipeline=None, count=6408, settings=None, ingest_fn=None):
    """假索引计数 + 假流水线工厂，TestClient 以 with 方式启动以触发 lifespan。"""
    original = api.collection_count
    api.collection_count = lambda *args, **kwargs: count
    try:
        app = api.create_app(
            pipeline_factory=lambda _settings: pipeline,
            settings=settings or _settings(),
            ingest_fn=ingest_fn,
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
        "bm25_built_at": "2025-10-09T08:53:20+00:00", "bm25_chunk_count": 6408,
        "device": "cpu", "mode": "hybrid", "llm_configured": True, "vector_store": "chroma",
        "llm_protocol": "openai", "llm_model": "test-model",
        "agent_ready": False,  # 测试配置没有 base_url -> 大模型未配齐 -> 对话 Agent 未启用
        "auth_required": False,  # 未设置 RAG_API_KEY
    }, body


def test_health_reports_configured_vector_store():
    with _client(_pipeline(), settings=_settings(vector_store="qdrant")) as client:
        assert client.get("/health").json()["vector_store"] == "qdrant"


def test_health_without_llm_and_bm25():
    with _client(_pipeline(chain=False, retriever=_Retriever(bm25_ready=False))) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok" and body["llm_configured"] is False and body["bm25_ready"] is False
    assert body["bm25_built_at"] is None and body["bm25_chunk_count"] is None


def test_health_not_ready_when_index_missing():
    with _client(_pipeline(), count=None) as client:  # 索引不存在：不建流水线，但服务能启动
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "not_ready" and body["chunk_count"] is None, body
        assert body["bm25_ready"] is False and body["bm25_built_at"] is None
        for path in ("/search", "/ask"):
            response = client.post(path, json={"query": "什么是软件工程"})
            body = _assert_error(response, 503, "service_unavailable")
            assert body["detail"] == "索引不存在或为空，请先运行 ragbase ingest --full"


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
    assert set(first) == {"rank", "chunk_id", "text", "source", "page", "heading_path", "scores"}
    assert first["heading_path"] == "第1章 概述 > 1.1 范围" and second["heading_path"] is None  # 旧版风格片段：null
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
    assert set(candidates[0]) == {"rank", "chunk_id", "text", "source", "page", "heading_path", "scores"}


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
    assert body["citations"][0]["heading_path"] == "第1章 概述 > 1.1 范围" and body["citations"][1]["heading_path"] is None
    assert body["citations"][0]["scores"]["rerank_score"] == 0.99
    context = pipeline.answer_chain.calls[0]["context"]
    # 有章节路径的片段在编号后带上章节；没有的保持原格式
    assert "[片段1]（第1章 概述 > 1.1 范围）片段文本0" in context and "[片段2] 片段文本1" in context
    assert "片段文本2" not in context
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


def test_ask_llm_timeout_returns_504_without_leaking():
    import httpx

    pipeline = _pipeline(error=httpx.ReadTimeout("read timed out sk-ULTRA-SECRET-999"))
    with _client(pipeline) as client:
        response = client.post("/ask", json={"query": "q"})
        # 超时之外的失败仍是 502
        pipeline.answer_chain.error = ConnectionError("连接被拒绝")
        other = client.post("/ask", json={"query": "q"})
    body = _assert_error(response, 504, "llm_timeout")
    assert "超时" in body["detail"] and "sk-ULTRA" not in response.text and "ReadTimeout" not in response.text
    _assert_error(other, 502, "llm_error")


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
    logging.getLogger("ragbase.api.app").addHandler(handler)
    try:
        with _client(_pipeline(retriever=retriever)) as client:
            response = client.post("/search", json={"query": "q"}, headers={"X-Request-ID": "boom-1"})
    finally:
        logging.getLogger("ragbase.api.app").removeHandler(handler)
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
    logger = logging.getLogger("ragbase.api.app")
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
    saved = os.environ.get("RAG_LLM_API_KEY")
    os.environ["RAG_LLM_API_KEY"] = secret
    get_settings.cache_clear()
    try:
        settings = get_settings()
        real_key = settings.llm_api_key.get_secret_value()
        assert real_key == secret  # 环境变量优先于 .env
        for text in (repr(settings), str(settings), settings.model_dump_json(), str(settings.model_dump())):
            assert secret not in text, text
        assert "SecretStr" in repr(settings) or "**********" in repr(settings)
        assert settings.llm_configured in (True, False)
    finally:
        if saved is None:
            os.environ.pop("RAG_LLM_API_KEY", None)
        else:
            os.environ["RAG_LLM_API_KEY"] = saved
        get_settings.cache_clear()
    health_settings = _settings(llm_api_key=secret, llm_model="m", llm_base_url="http://x")
    with _client(_pipeline(), settings=health_settings) as client:
        assert secret not in client.get("/health").text


# ---------------------------------------------------------------------------
# 文档管理、入库任务、按书过滤
# ---------------------------------------------------------------------------

class _FakeIngest:
    """假的入库函数：发出几条进度事件，可用 gate 卡住以测试“任务运行中”的行为。"""

    def __init__(self, gate=None, error=None, failed_files=None, changed=True):
        self.gate = gate
        self.error = error
        self.failed_files = failed_files or {}
        self.changed = changed
        self.started = threading.Event()
        self.calls = []

    def __call__(self, pipeline, settings, progress):
        self.calls.append(settings.data_dir)
        self.started.set()
        progress("plan", reason=None, to_add=["a.md", "b.md"], deleted=["c.md"], unchanged=1, skipped=["old.doc"])
        if self.gate is not None:
            assert self.gate.wait(10), "测试门闩超时"
        if self.error:
            raise self.error
        progress("file_start", name="a.md")
        progress("file_done", name="a.md", chunks_added=3, chunks_deleted=1, report={})
        progress("file_deleted", name="c.md", chunks_deleted=2)
        for name, reason in self.failed_files.items():
            progress("file_failed", name=name, error=reason)
        return IngestResult(
            files_total=3, files_done=3, chunks_added=3, chunks_deleted=3,
            failed_files=self.failed_files, changed=self.changed,
        )


def _ingest_settings(tmp, **overrides):
    root = Path(tmp)
    values = dict(
        data_dir=root / "data", db_dir=root / "chroma", meta_file=root / "meta.json",
        cache_dir=root / "cache", max_upload_mb=1,
    )
    return _settings(**{**values, **overrides})


def _upload(client, name, data=b"# \xe6\xa0\x87\xe9\xa2\x98\n\n\xe6\xad\xa3\xe6\x96\x87", content_type="application/octet-stream"):
    return client.post("/documents", files={"file": (name, data, content_type)})


def _wait_job(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/ingest/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError("入库任务超时未结束")


JOB_KEYS = {
    "job_id", "status", "files_total", "files_done", "current_file", "chunks_added", "chunks_deleted",
    "skipped_files", "error", "started_at", "finished_at",
}


def test_documents_list_comes_from_retriever_catalog():
    with _client(_pipeline()) as client:
        response = client.get("/documents")
    assert response.status_code == 200, response.text
    assert response.json() == {"documents": [
        {"name": "book.pdf", "chunk_count": 10, "heading_chunks": 4, "has_headings": True},
        {"name": "other.pdf", "chunk_count": 10, "heading_chunks": 0, "has_headings": False},
    ]}


def test_document_endpoints_return_503_when_index_missing():
    with _client(_pipeline(), count=None) as client:
        _assert_error(client.get("/documents"), 503, "service_unavailable")
        _assert_error(client.post("/ingest"), 503, "service_unavailable")
        _assert_error(_upload(client, "a.md"), 503, "service_unavailable")
        _assert_error(client.delete("/documents/a.md"), 503, "service_unavailable")


def test_upload_rejects_bad_extension_legacy_doc_and_bad_names():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _ingest_settings(tmp)
        ingest = _FakeIngest()
        with _client(_pipeline(), settings=settings, ingest_fn=ingest) as client:
            body = _assert_error(_upload(client, "virus.exe"), 400, "bad_request")
            assert ".exe" in body["detail"] and ".pdf" in body["detail"]
            body = _assert_error(_upload(client, "old.DOC"), 400, "bad_request")
            assert ".docx" in body["detail"]
            _assert_error(_upload(client, "noext"), 400, "bad_request")
            for bad in (".hidden.md", "~$lock.docx", "README.md", "readme.TXT", ".."):
                _assert_error(_upload(client, bad), 400, "bad_request")
            body = _assert_error(_upload(client, "empty.md", b""), 400, "bad_request")
            assert "为空" in body["detail"]
            _assert_error(client.post("/documents"), 422, "validation_error")  # 缺 file 字段
        assert ingest.calls == []  # 被拒绝的上传不会启动入库
        assert not list(settings.data_dir.glob("*")) if settings.data_dir.exists() else True
        assert not list(settings.data_dir.glob(".*")) if settings.data_dir.exists() else True  # 临时文件也已清理


def test_upload_rejects_oversized_files():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _ingest_settings(tmp)  # 上限 1 MB
        with _client(_pipeline(), settings=settings, ingest_fn=_FakeIngest()) as client:
            # 明显超限：Content-Length 阶段就拒绝
            body = _assert_error(_upload(client, "big.md", b"x" * (2 * 1024 * 1024)), 413, "payload_too_large")
            assert "1 MB" in body["detail"]
            # 刚过上限但在 multipart 开销容差内：写盘时按精确字节数拒绝
            _assert_error(_upload(client, "edge.md", b"x" * (1024 * 1024 + 10)), 413, "payload_too_large")
            ok = _upload(client, "fits.md", b"x" * (1024 * 1024))  # 正好等于上限：允许
            assert ok.status_code == 202, ok.text
        names = sorted(path.name for path in settings.data_dir.iterdir())
        assert names == ["fits.md"], names  # 超限文件和临时文件都没留下


def test_upload_filename_is_reduced_to_basename_and_stays_inside_data_dir():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _ingest_settings(tmp)
        with _client(_pipeline(), settings=settings, ingest_fn=_FakeIngest()) as client:
            for index, raw in enumerate(("../../evil.md", "..\\..\\evil2.md", "/abs/path/evil3.txt")):
                response = _upload(client, raw)
                assert response.status_code == 202, (raw, response.text)
                _wait_job(client, response.json()["job_id"])
        assert sorted(path.name for path in settings.data_dir.iterdir()) == ["evil.md", "evil2.md", "evil3.txt"]
        assert not (Path(tmp) / "evil.md").exists() and not (Path(tmp).parent / "evil.md").exists()


def test_upload_success_starts_job_that_runs_to_completion_and_refreshes_bm25():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _ingest_settings(tmp)
        ingest = _FakeIngest()
        pipeline = _pipeline()
        with _client(pipeline, settings=settings, ingest_fn=ingest) as client:
            response = _upload(client, "Notes.MD", "你好".encode("utf-8"))
            assert response.status_code == 202, response.text
            accepted = response.json()
            assert set(accepted) == JOB_KEYS | {"filename"} and accepted["filename"] == "Notes.MD"
            assert accepted["status"] in ("pending", "running", "succeeded")
            job = _wait_job(client, accepted["job_id"])
            assert set(job) == JOB_KEYS
            assert job["status"] == "succeeded" and job["error"] is None and job["current_file"] is None
            assert (job["files_total"], job["files_done"]) == (3, 2)  # 计划 2 新增 + 1 删除；事件里完成了 a.md 和 c.md
            assert (job["chunks_added"], job["chunks_deleted"]) == (3, 3)
            assert job["skipped_files"] == ["old.doc"]
            assert job["started_at"] and job["finished_at"]
            assert pipeline.retriever.refreshes == 1  # 任务结束后刷新了 BM25
            # 同名文件再次上传：覆盖
            again = _upload(client, "Notes.MD", "新内容".encode("utf-8"))
            assert again.status_code == 202
            _wait_job(client, again.json()["job_id"])
            _assert_error(client.get("/ingest/jobs/nope"), 404, "not_found")
        assert (settings.data_dir / "Notes.MD").read_text(encoding="utf-8") == "新内容"
        assert ingest.calls == [settings.data_dir, settings.data_dir]
        assert not [path for path in settings.data_dir.iterdir() if path.name.startswith(".")]


def test_only_one_job_at_a_time_and_upload_during_job_keeps_file():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _ingest_settings(tmp)
        gate = threading.Event()
        ingest = _FakeIngest(gate=gate)
        pipeline = _pipeline()
        with _client(pipeline, settings=settings, ingest_fn=ingest) as client:
            first = client.post("/ingest")
            assert first.status_code == 202 and set(first.json()) == JOB_KEYS
            job_id = first.json()["job_id"]
            assert ingest.started.wait(5)
            running = client.get(f"/ingest/jobs/{job_id}").json()
            assert running["status"] == "running" and running["files_total"] == 3  # 进度已可见
            body = _assert_error(client.post("/ingest"), 409, "conflict")
            assert job_id in body["detail"]
            body = _assert_error(_upload(client, "late.md"), 409, "conflict")
            assert "late.md" in body["detail"] and "POST /ingest" in body["detail"]
            assert (settings.data_dir / "late.md").is_file()  # 文件已保存
            _assert_error(client.delete("/documents/book.pdf"), 409, "conflict")
            gate.set()
            assert _wait_job(client, job_id)["status"] == "succeeded"
            again = client.post("/ingest")  # 看到 succeeded 后立刻就能启动下一个任务
            assert again.status_code == 202, again.text
            assert _wait_job(client, again.json()["job_id"])["status"] == "succeeded"
        assert len(ingest.calls) == 2


def test_job_failure_paths():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _ingest_settings(tmp)
        logging.getLogger("ragbase.ingest.jobs").disabled = True  # 预期内的失败，不在测试输出里刷堆栈
        logging.getLogger("ragbase.api.app").disabled = True
        try:
            # 入库函数抛异常：任务 failed，但仍尝试刷新 BM25（索引可能已部分改动）
            pipeline = _pipeline()
            with _client(pipeline, settings=settings, ingest_fn=_FakeIngest(error=RuntimeError("磁盘满了"))) as client:
                job = _wait_job(client, client.post("/ingest").json()["job_id"])
                assert job["status"] == "failed" and "磁盘满了" in job["error"] and job["finished_at"]
                assert pipeline.retriever.refreshes == 1
                # 之后仍可启动新任务（忙标记已释放）
            # 部分文件失败：整体 failed，失败文件列入 skipped_files，其余已入库并刷新
            pipeline = _pipeline()
            ingest = _FakeIngest(failed_files={"bad.docx": "BadZipFile: x"})
            with _client(pipeline, settings=settings, ingest_fn=ingest) as client:
                job = _wait_job(client, client.post("/ingest").json()["job_id"])
                assert job["status"] == "failed" and "bad.docx" in job["error"] and "其余文件已入库" in job["error"]
                assert "bad.docx（入库失败）" in job["skipped_files"] and "old.doc" in job["skipped_files"]
                assert pipeline.retriever.refreshes == 1
                assert client.post("/ingest").status_code == 202
            # 没有变化：不刷新 BM25
            pipeline = _pipeline()
            with _client(pipeline, settings=settings, ingest_fn=_FakeIngest(changed=False)) as client:
                assert _wait_job(client, client.post("/ingest").json()["job_id"])["status"] == "succeeded"
                assert pipeline.retriever.refreshes == 0
            # 刷新失败：入库已完成，但任务标记 failed 并说明原因
            pipeline = _pipeline()
            pipeline.retriever.fail_refresh = True
            with _client(pipeline, settings=settings, ingest_fn=_FakeIngest()) as client:
                job = _wait_job(client, client.post("/ingest").json()["job_id"])
                assert job["status"] == "failed" and "BM25" in job["error"]
        finally:
            logging.getLogger("ragbase.ingest.jobs").disabled = False
            logging.getLogger("ragbase.api.app").disabled = False


class _VectorEmbeddings:
    """确定性假 embedding（8 维），同时给入库（embed_documents）和检索（embed_query）用。"""

    @staticmethod
    def _vector(text):
        import hashlib

        return [byte / 255 for byte in hashlib.md5(text.encode("utf-8")).digest()[:8]]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


class _AllReranker:
    def rerank_with_scores(self, query, documents, top_k=5):
        return [(index, 0.9 - index * 0.01) for index in range(min(top_k, len(documents)))]


def _real_pipeline(tmp, legacy_chunks=1):
    """真实 HybridRetriever + 临时 ChromaStore + 假 embedding：用来端到端验证上传 / 过滤 / 删除。"""
    from ragbase.retrieval.hybrid import HybridRetriever
    from ragbase.vectorstores.chroma import ChromaStore

    embeddings = _VectorEmbeddings()
    collection = ChromaStore(Path(tmp) / "chroma", "rag")
    for index in range(legacy_chunks):  # 旧版索引风格：UUID 形式 ID，无 fid / 章节，source 为别处的绝对路径
        collection.upsert(
            ids=[f"00000000-0000-0000-0000-{index:012d}"],
            embeddings=[embeddings.embed_query(f"旧{index}")],
            documents=[f"旧版片段{index}"],
            metadatas=[{"source": "/legacy/dir/old.pdf", "page": index}],
        )
    retriever = HybridRetriever(collection, embeddings, mode="hybrid", cache_dir=Path(tmp) / "cache")
    return RAGPipeline(
        retriever=retriever, reranker=_AllReranker(), answer_chain=_Chain([]), embeddings=embeddings, device="cpu",
    ), collection


def test_end_to_end_upload_filter_and_delete_on_a_legacy_index():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _ingest_settings(tmp, max_upload_mb=5)
        pipeline, collection = _real_pipeline(tmp, legacy_chunks=2)
        with _client(pipeline, count=2, settings=settings) as client:  # 默认 ingest_fn：真实的 run_ingest + 假 embedding
            listing = client.get("/documents").json()["documents"]
            assert listing == [{"name": "old.pdf", "chunk_count": 2, "heading_chunks": 0, "has_headings": False}]
            before = client.get("/health").json()
            assert before["bm25_chunk_count"] == 2

            text = "\n\n".join(f"关键路径第{i}段：" + "关键路径法的内容" * 20 for i in range(6))
            job = _wait_job(client, _upload(client, "first.md", text.encode("utf-8")).json()["job_id"])
            assert job["status"] == "succeeded", job
            assert job["chunks_deleted"] == 2 and job["chunks_added"] > 0  # 旧版索引先按 ID 清空，再写入新文件
            names = [entry["name"] for entry in client.get("/documents").json()["documents"]]
            assert names == ["first.md"]  # 旧版片段已迁移走
            health = client.get("/health").json()
            assert health["bm25_chunk_count"] == job["chunks_added"] and health["bm25_built_at"] >= before["bm25_built_at"]  # 秒级精度，可能同一秒

            second = "\n\n".join(f"项目沟通第{i}段：" + "沟通管理计划的内容" * 20 for i in range(4))
            job2 = _wait_job(client, _upload(client, "second.txt", second.encode("utf-8")).json()["job_id"])
            assert job2["status"] == "succeeded" and job2["chunks_deleted"] == 0 and job2["files_total"] == 1
            docs = {entry["name"]: entry["chunk_count"] for entry in client.get("/documents").json()["documents"]}
            assert set(docs) == {"first.md", "second.txt"} and docs["first.md"] == job["chunks_added"]

            # 按书过滤：/search 只返回所选书的片段，未知书名 400
            for mode in ("vector", "bm25", "hybrid"):
                found = client.post("/search", json={"query": "沟通管理计划", "mode": mode, "sources": ["second.txt"]}).json()
                assert found["chunks"] and {chunk["source"] for chunk in found["chunks"]} == {"second.txt"}, (mode, found)
                assert all(chunk["heading_path"] is None for chunk in found["chunks"])
            unknown = _assert_error(client.post("/search", json={"query": "q", "sources": ["nope.pdf"]}), 400, "bad_request")
            assert "nope.pdf" in unknown["detail"] and "first.md" in unknown["detail"] and "second.txt" in unknown["detail"]
            # BM25 路过滤后零命中（查询词只在另一本书里）-> 没有召回，/search 判定拒答
            empty = client.post("/search", json={"query": "关键路径法", "mode": "bm25", "sources": ["second.txt"]}).json()
            assert empty["chunks"] == [] and empty["refused"] is True and empty["top_score"] is None

            # 删除：片段、文件、元数据、BM25 都更新
            from ragbase.ingest.common import load_ingest_meta

            assert {info["name"] for info in load_ingest_meta(settings.meta_file).values()} == {"first.md", "second.txt"}
            deleted = client.delete("/documents/second.txt")
            assert deleted.status_code == 200 and deleted.json() == {"name": "second.txt", "chunks_deleted": docs["second.txt"]}
            assert not (settings.data_dir / "second.txt").exists()
            assert [info["name"] for info in load_ingest_meta(settings.meta_file).values()] == ["first.md"]
            assert [entry["name"] for entry in client.get("/documents").json()["documents"]] == ["first.md"]
            assert client.get("/health").json()["bm25_chunk_count"] == docs["first.md"]
            _assert_error(client.delete("/documents/second.txt"), 404, "not_found")
            _assert_error(client.delete("/documents/.hidden"), 404, "not_found")
            _assert_error(client.post("/search", json={"query": "q", "sources": ["second.txt"]}), 400, "bad_request")


def test_delete_legacy_document_by_source_basename_and_unindexed_file():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _ingest_settings(tmp)
        pipeline, collection = _real_pipeline(tmp, legacy_chunks=3)
        settings.data_dir.mkdir(parents=True)
        (settings.data_dir / "pending.md").write_text("还没入库", encoding="utf-8")
        with _client(pipeline, count=3, settings=settings) as client:
            # 旧版片段没有 fid：按 source 的 basename 匹配（库里存的是别处的绝对路径）
            response = client.delete("/documents/old.pdf")
            assert response.status_code == 200 and response.json() == {"name": "old.pdf", "chunks_deleted": 3}
            assert collection.count() == 0
            assert client.get("/documents").json() == {"documents": []}
            # 只存在于数据目录、还没入库的文件：删文件，片段数 0
            response = client.delete("/documents/pending.md")
            assert response.status_code == 200 and response.json() == {"name": "pending.md", "chunks_deleted": 0}
            assert not (settings.data_dir / "pending.md").exists()
            _assert_error(client.delete("/documents/pending.md"), 404, "not_found")


def test_sources_passed_through_search_and_ask_and_validated():
    pipeline = _pipeline()
    with _client(pipeline) as client:
        found = client.post("/search", json={"query": "q", "sources": ["book.pdf", "book.pdf"]})
        assert found.status_code == 200, found.text
        assert [chunk["source"] for chunk in found.json()["chunks"]] == ["book.pdf", "book.pdf"]
        assert pipeline.retriever.source_calls[-1] == ["book.pdf"]  # 去重后传给检索器
        client.post("/search", json={"query": "q"})
        client.post("/search", json={"query": "q", "sources": []})
        assert pipeline.retriever.source_calls[-2:] == [None, None]  # 缺省 / 空列表 = 不过滤
        asked = client.post("/ask", json={"query": "q", "sources": ["other.pdf"], "context_min_score": 0})
        assert asked.status_code == 200, asked.text
        assert pipeline.retriever.source_calls[-1] == ["other.pdf"]
        assert [c["source"] for c in asked.json()["citations"]] == ["other.pdf"]
        for path in ("/search", "/ask"):
            body = _assert_error(client.post(path, json={"query": "q", "sources": ["book.pdf", "nope.pdf"]}), 400, "bad_request")
            assert "nope.pdf" in body["detail"] and "book.pdf" in body["detail"] and "other.pdf" in body["detail"]
        _assert_error(client.post("/search", json={"query": "q", "sources": "book.pdf"}), 422, "validation_error")
        _assert_error(client.post("/search", json={"query": "q", "sources": ["x"] * 51}), 422, "validation_error")
    calls_before_unknown = len(pipeline.retriever.calls)
    assert calls_before_unknown == 4  # 两次未知书名的请求在检索前就被拒绝，没有走到检索器


def test_sources_filter_with_no_matching_chunks_is_refused():
    pipeline = _pipeline(scores=[])
    with _client(pipeline) as client:
        body = client.post("/search", json={"query": "q", "sources": ["other.pdf"]}).json()
        asked = client.post("/ask", json={"query": "q", "sources": ["other.pdf"]}).json()
    assert body["chunks"] == [] or body["refused"] is True
    assert asked["refused"] is True and asked["answer"] == NO_ANSWER and asked["citations"] == []


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
