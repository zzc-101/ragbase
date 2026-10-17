"""静态 API Key 鉴权的单元测试：TestClient + 假流水线 / 脚本化假模型（不加载模型、不访问网络、不调用真实大模型）。

覆盖：鉴权关闭时行为不变；开启后每个受保护接口缺密钥 / 错密钥（两种请求头）都是 401 且带统一错误 JSON 与 WWW-Authenticate，
正确密钥放行；/health 精简 vs 完整；前端静态资源 / SPA / 文档公开；新增路由默认受保护；SSE 与大文件上传在处理前就被拒绝；
密钥不出现在日志 / 错误体里；CORS 仅在配置时启用；非回环地址且无密钥时的启动警告；配置解析。

运行：uv run python tests/test_auth.py
"""
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient

from ragbase.api import app as api
from ragbase.api.auth import is_loopback_host
from ragbase.config import Settings
from ragbase.ingest.service import IngestResult
from test_agent import ScriptedModel, _pipeline, _settings, build_chat_service, say, search

logging.disable(logging.NOTSET)  # test_agent 导入时关掉了日志；本测试要捕获日志

_SCRATCH = tempfile.TemporaryDirectory(prefix="ragdemo-auth-test-")
SCRATCH = Path(_SCRATCH.name)
KEY = "test-key-123"
WRONG = "wrong-key-XYZ"
INDEX_HTML = "<!doctype html><title>RAG 知识库</title><div id=root></div>"

HEALTH_KEYS = {
    "status", "chunk_count", "bm25_ready", "bm25_built_at", "bm25_chunk_count", "device", "mode",
    "llm_configured", "vector_store", "llm_protocol", "llm_model", "agent_ready", "auth_required",
    "embedding", "reranker",
}


def _fake_ingest(pipeline, settings, progress):
    return IngestResult(files_total=0, changed=False)


@contextmanager
def _client(model=None, api_key=KEY, **overrides):
    original = api.collection_count
    api.collection_count = lambda *args, **kwargs: 100
    pipeline = _pipeline()
    dist = SCRATCH / "dist"
    (dist / "assets").mkdir(parents=True, exist_ok=True)
    (dist / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    values = {"web_dist": dist, "max_upload_mb": 1, **overrides}
    if api_key is not None:
        values["api_key"] = api_key
    model = model or ScriptedModel(script=[say("答")], repeat=True)
    try:
        app = api.create_app(
            pipeline_factory=lambda _s: pipeline,
            settings=_settings(**values),
            ingest_fn=_fake_ingest,
            chat_factory=lambda p, s: build_chat_service(p, s, model=model),
        )
        with TestClient(app) as client:
            client.model = model
            yield client
    finally:
        api.collection_count = original


def _bearer(key=KEY):
    return {"Authorization": f"Bearer {key}"}


def _xkey(key=KEY):
    return {"X-API-Key": key}


# 受保护的接口：(方法, 路径, 请求参数, 带正确密钥时的期望状态码)
def _protected_requests():
    upload = {"files": {"file": ("bad.exe", b"x", "application/octet-stream")}}
    return [
        ("POST", "/search", {"json": {"query": "q"}}, 200),
        ("POST", "/ask", {"json": {"query": "q"}}, 503),  # 假流水线没有大模型链 -> 503，说明已经越过鉴权
        ("POST", "/chat", {"json": {"message": "你好", "session_id": "auth1"}}, 200),
        ("POST", "/chat/stream", {"json": {"message": "你好", "session_id": "auth2"}}, 200),
        ("GET", "/chat/sessions", {}, 200),
        ("GET", "/chat/sessions/auth1", {}, 200),
        ("DELETE", "/chat/sessions/auth1", {}, 200),
        ("GET", "/documents", {}, 200),
        ("POST", "/documents", upload, 400),  # 不支持的扩展名 -> 400（越过鉴权）
        ("DELETE", "/documents/none.pdf", {}, 404),
        ("POST", "/ingest", {}, 202),
        ("GET", "/ingest/jobs/nope", {}, 404),
    ]


def _assert_unauthorized(response):
    assert response.status_code == 401, (response.status_code, response.text)
    body = response.json()
    assert set(body) == {"error", "detail"} and body["error"] == "unauthorized" and body["detail"], body
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.headers.get("x-request-id")
    for secret in (KEY, WRONG):
        assert secret not in response.text and secret not in str(response.headers)


# ---------------------------------------------------------------- 测试
def test_disabled_mode_is_unchanged():
    with _client(api_key=None) as client:
        for method, path, kwargs, _ in _protected_requests()[:5]:
            response = client.request(method, path, **kwargs)
            assert response.status_code != 401, (path, response.text)
        health = client.get("/health").json()
        assert set(health) == HEALTH_KEYS and health["auth_required"] is False and health["status"] == "ok"
        assert client.post("/search", json={"query": "q"}).status_code == 200
        # 即使带了密钥头也无妨
        assert client.post("/search", json={"query": "q"}, headers=_bearer(WRONG)).status_code == 200
    with _client(api_key="   ") as client:  # 空白 = 未设置
        assert client.get("/health").json()["auth_required"] is False
        assert client.get("/documents").status_code == 200


def test_every_protected_route_requires_key_in_both_header_styles():
    with _client() as client:
        for method, path, kwargs, expected in _protected_requests():
            label = f"{method} {path}"
            for headers in (None, _bearer(WRONG), _xkey(WRONG), {"Authorization": f"Basic {KEY}"}, {"Authorization": KEY}):
                _assert_unauthorized(client.request(method, path, headers=headers, **kwargs))
            for headers in (_bearer(), _xkey(), {"Authorization": f"bearer {KEY}"}, {**_bearer(WRONG), **_xkey()}):
                response = client.request(method, path, headers=headers, **kwargs)
                if label == "DELETE /chat/sessions/auth1":
                    continue  # 会话只能删一次，单独断言
                if label.startswith("GET /chat/sessions/auth1"):
                    assert response.status_code in (200, 404), (label, response.text)
                    continue
                assert response.status_code == expected, (label, headers, response.status_code, response.text)
        # 会话接口：先建会话，再用密钥读 / 删
        assert client.post("/chat", json={"message": "x", "session_id": "auth9"}, headers=_bearer()).status_code == 200
        assert client.get("/chat/sessions/auth9", headers=_xkey()).status_code == 200
        _assert_unauthorized(client.delete("/chat/sessions/auth9"))
        assert client.delete("/chat/sessions/auth9", headers=_bearer()).status_code == 200


def test_unauthenticated_validation_error_does_not_leak_before_auth():
    with _client() as client:
        _assert_unauthorized(client.post("/search", json={}))  # 无密钥时连 422 都不给，先 401
        assert client.post("/search", json={}, headers=_bearer()).status_code == 422


def test_non_ascii_key_header_is_rejected_cleanly():
    with _client() as client:
        response = client.get("/documents", headers={"X-API-Key": "密钥".encode("utf-8")})
        _assert_unauthorized(response)
    with _client(api_key="密钥-123") as client:  # 配置里的密钥含非 ASCII 也能匹配（按 UTF-8 字节比较）
        ok = client.get("/documents", headers={"X-API-Key": "密钥-123".encode("utf-8")})
        assert ok.status_code == 200


def test_health_minimal_without_key_and_full_with_key():
    with _client() as client:
        for headers in (None, _bearer(WRONG), _xkey(WRONG)):
            response = client.get("/health", headers=headers)
            assert response.status_code == 200
            assert response.json() == {"status": "ok", "auth_required": True}, response.json()
        for headers in (_bearer(), _xkey()):
            body = client.get("/health", headers=headers).json()
            assert set(body) == HEALTH_KEYS and body["auth_required"] is True and body["chunk_count"] == 100, body


def test_frontend_and_docs_are_public():
    with _client() as client:
        for path in ("/", "/library", "/c/abc123", "/docs", "/redoc", "/openapi.json"):
            response = client.get(path)
            assert response.status_code == 200, (path, response.status_code)
        assert client.get("/").text == INDEX_HTML
        assert client.get("/assets/missing.js").status_code == 404  # 越过鉴权，由静态路由返回 404
        assert KEY not in client.get("/openapi.json").text
        # 公开的只有 GET / HEAD 的前端路径，其他仍受保护
        _assert_unauthorized(client.post("/"))
        _assert_unauthorized(client.get("/c/a/b"))
        _assert_unauthorized(client.get("/assets/"))
        _assert_unauthorized(client.get("/healthz"))
        _assert_unauthorized(client.get("/not-a-route"))  # 未知路径也先 401，不泄露哪些路由存在
        _assert_unauthorized(client.post("/health"))


def test_new_routes_are_protected_by_default():
    with _client() as client:
        client.app.add_api_route("/brand-new", lambda: {"secret": 1}, methods=["GET"])
        _assert_unauthorized(client.get("/brand-new"))
        assert client.get("/brand-new", headers=_bearer()).json() == {"secret": 1}


def test_sse_and_oversized_upload_are_rejected_before_processing():
    model = ScriptedModel(script=[search("q"), say("答")], repeat=True)
    with _client(model) as client:
        response = client.post("/chat/stream", json={"message": "你好"})
        _assert_unauthorized(response)
        assert response.headers["content-type"].startswith("application/json")  # 不是 text/event-stream
        assert model.seen == []  # 模型没有被调用
        # 超过上限的上传：未鉴权时是 401，而不是 413（鉴权先于任何体积检查，且不读请求体）
        big = {"file": ("big.pdf", b"0" * (2 * 1024 * 1024), "application/pdf")}
        _assert_unauthorized(client.post("/documents", files=big))
        assert client.post("/documents", files=big, headers=_bearer()).status_code == 413
        # 带密钥的流式请求正常
        ok = client.post("/chat/stream", json={"message": "你好"}, headers=_bearer())
        assert ok.status_code == 200 and ok.headers["content-type"].startswith("text/event-stream")
        assert "event: done" in ok.text


def test_key_never_appears_in_logs():
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(self.format(record))

    handler = _Capture(level=logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(name)s %(message)s"))
    with _client(log_level="DEBUG") as client:  # create_app 会重置根 logger 的 handler，所以在其后挂
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            client.get("/documents")
            client.get("/documents", headers=_bearer(WRONG))
            client.get("/documents", headers=_xkey(WRONG))
            client.get("/documents", headers=_bearer())
            client.post("/search", json={"query": "q"}, headers=_xkey())
            client.get("/health", headers=_bearer())
        finally:
            root.removeHandler(handler)
    text = "\n".join(records)
    assert "GET /documents -> 401" in text and "POST /search -> 200" in text, text  # 确实捕获到了访问日志
    assert KEY not in text and WRONG not in text, text


def test_cors_only_when_configured():
    origin = "https://app.example.com"
    with _client() as client:  # 未配置：没有 CORS 头
        response = client.get("/documents", headers={"Origin": origin})
        assert "access-control-allow-origin" not in response.headers
        preflight = client.options("/documents", headers={"Origin": origin, "Access-Control-Request-Method": "GET"})
        assert "access-control-allow-origin" not in preflight.headers
    with _client(cors_origins=f"{origin}, http://localhost:5173/") as client:
        assert client.app.state.settings.cors_origins == [origin, "http://localhost:5173"]
        # 预检不需要密钥，并允许 Authorization / X-API-Key / Content-Type
        preflight = client.options(
            "/chat/stream",
            headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                     "Access-Control-Request-Headers": "authorization,x-api-key,content-type"},
        )
        assert preflight.status_code == 200, preflight.text
        assert preflight.headers["access-control-allow-origin"] == origin
        allowed = preflight.headers["access-control-allow-headers"].lower()
        assert all(name in allowed for name in ("authorization", "x-api-key", "content-type")), allowed
        # 401 响应同样带 CORS 头，浏览器才能读到错误；并暴露 X-Request-ID
        denied = client.get("/documents", headers={"Origin": origin})
        _assert_unauthorized(denied)
        assert denied.headers["access-control-allow-origin"] == origin
        assert "x-request-id" in denied.headers["access-control-expose-headers"].lower()
        ok = client.get("/documents", headers={"Origin": origin, **_bearer()})
        assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == origin
        # 未列出的来源不放行
        other = client.get("/documents", headers={"Origin": "https://evil.example.com", **_bearer()})
        assert "access-control-allow-origin" not in other.headers


def test_startup_warning_on_exposed_host_without_key():
    class _Capture(logging.Handler):
        def __init__(self):
            super().__init__(logging.WARNING)
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    original_run, original_settings = api.uvicorn.run, api.get_settings
    calls = []
    api.uvicorn.run = lambda app, **kwargs: calls.append(kwargs)
    logger = logging.getLogger("ragbase.api.app")
    try:
        def serve(host, api_key=None, settings_host="127.0.0.1"):
            settings = Settings(_env_file=None, host=settings_host, **({"api_key": api_key} if api_key else {}))
            api.get_settings = lambda: settings
            handler = _Capture()
            logger.addHandler(handler)
            try:
                api.serve(host=host, port=8999)
            finally:
                logger.removeHandler(handler)
            return handler.messages

        warning = "未设置 RAG_API_KEY，服务对外暴露且没有鉴权"
        assert serve("0.0.0.0") == [warning]
        assert serve("::") == [warning]
        assert serve("192.168.1.5") == [warning]
        assert serve(None, settings_host="0.0.0.0") == [warning]  # 取配置里的 RAG_HOST
        assert serve("0.0.0.0", api_key=KEY) == []
        for quiet in ("127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2", None):
            assert serve(quiet) == [], quiet
        assert calls[-1]["host"] == "127.0.0.1" and calls[0]["host"] == "0.0.0.0"
    finally:
        api.uvicorn.run, api.get_settings = original_run, original_settings
    assert is_loopback_host("LOCALHOST") and not is_loopback_host("0.0.0.0") and not is_loopback_host("example.com")


def test_settings_parsing():
    saved = {k: os.environ.pop(k) for k in list(os.environ) if k in ("RAG_API_KEY", "RAG_CORS_ORIGINS")}
    try:
        settings = Settings(_env_file=None)
        assert settings.api_key is None and settings.cors_origins == []
        os.environ["RAG_API_KEY"] = f"  {KEY}  "
        os.environ["RAG_CORS_ORIGINS"] = " https://a.example.com/ ,http://localhost:5173,, "
        settings = Settings(_env_file=None)
        assert settings.api_key.get_secret_value() == KEY  # 去掉首尾空白
        assert settings.cors_origins == ["https://a.example.com", "http://localhost:5173"]
        assert KEY not in repr(settings) and KEY not in str(settings.model_dump())
        os.environ["RAG_API_KEY"] = ""
        os.environ["RAG_CORS_ORIGINS"] = ""
        settings = Settings(_env_file=None)
        assert settings.api_key is None and settings.cors_origins == []
    finally:
        os.environ.pop("RAG_API_KEY", None)
        os.environ.pop("RAG_CORS_ORIGINS", None)
        os.environ.update(saved)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
