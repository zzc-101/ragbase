"""Web 前端托管与会话列表的单元测试：假流水线 + 脚本化假模型（不加载模型、不访问网络、不调用真实大模型）。

覆盖：GET /chat/sessions（排序、标题截断、消息数、删除后消失、Agent 未就绪返回空列表）、
静态资源与 SPA 回退（临时 dist 目录）、API 的 GET 路由不被前端路由遮住、dist 缺失时的提示。

运行：uv run python tests/test_web.py
"""
import logging
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient

from ragbase.api import app as api
from ragbase.agent.service import TITLE_MAX_CHARS
from test_agent import ScriptedModel, _pipeline, _settings, build_chat_service, say

logging.disable(logging.CRITICAL)

_SCRATCH = tempfile.TemporaryDirectory(prefix="ragdemo-web-test-")
SCRATCH = Path(_SCRATCH.name)
INDEX_HTML = "<!doctype html><title>RAG 知识库</title><div id=root></div>"


def _make_dist(name: str) -> Path:
    dist = SCRATCH / name
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (dist / "assets" / "app-abc123.js").write_text("console.log('hi')", encoding="utf-8")
    (SCRATCH / "secret.txt").write_text("不应被读到", encoding="utf-8")
    return dist


@contextmanager
def _client(model=None, factory=None, dist=None):
    original = api.collection_count
    api.collection_count = lambda *args, **kwargs: 100
    pipeline = _pipeline()
    kwargs = {"web_dist": dist} if dist is not None else {}
    chat_factory = factory or (lambda p, s: build_chat_service(p, s, model=model))
    try:
        app = api.create_app(pipeline_factory=lambda _s: pipeline, settings=_settings(**kwargs), chat_factory=chat_factory)
        with TestClient(app) as client:
            yield client
    finally:
        api.collection_count = original


def test_sessions_list():
    long_question = "这是一个很长很长的问题" * 10 + "\n第二行"
    model = ScriptedModel(script=[say("答一"), say("答二"), say("答三"), say("答四")])
    with _client(model) as client:
        assert client.get("/chat/sessions").json() == {"sessions": []}
        service = client.app.state.chat
        ticks = iter(range(1000, 2000))
        service.sessions.wall_clock = lambda: next(ticks) * 10.0  # 保证每次使用的挂钟时间严格递增
        assert client.post("/chat", json={"session_id": "s-old", "message": "最早的问题"}).status_code == 200
        assert client.post("/chat", json={"session_id": "s-long", "message": long_question}).status_code == 200
        assert client.post("/chat", json={"session_id": "s-old", "message": "追问"}).status_code == 200  # s-old 变成最近使用

        body = client.get("/chat/sessions").json()
        assert set(body) == {"sessions"}
        sessions = body["sessions"]
        assert [s["session_id"] for s in sessions] == ["s-old", "s-long"]  # 最近使用的在前
        old, long = sessions
        assert set(old) == {"session_id", "title", "updated_at", "message_count"}
        assert old["title"] == "最早的问题" and old["message_count"] == 4  # 2 问 + 2 答
        assert long["title"] == long_question.replace("\n", " ")[:TITLE_MAX_CHARS] + "…" and long["message_count"] == 2
        assert old["updated_at"] > long["updated_at"] and old["updated_at"].endswith("+00:00")

        # 删除后从列表消失；GET /chat/sessions 不遮住 /chat/sessions/{id}
        assert client.delete("/chat/sessions/s-old").status_code == 200
        assert [s["session_id"] for s in client.get("/chat/sessions").json()["sessions"]] == ["s-long"]
        assert client.get("/chat/sessions/s-long").json()["session_id"] == "s-long"


def test_sessions_hide_rolled_back_turns_and_tool_steps():
    def boom(messages):
        raise RuntimeError("upstream exploded")

    from test_agent import search

    model = ScriptedModel(script=[boom, search("A"), say("答[1]")])
    with _client(model) as client:
        assert client.post("/chat", json={"session_id": "bad", "message": "会失败"}).status_code == 502
        assert client.get("/chat/sessions").json() == {"sessions": []}  # 失败的首轮被回滚，不列出空会话
        assert client.post("/chat", json={"session_id": "ok", "message": "查资料"}).status_code == 200
        (item,) = client.get("/chat/sessions").json()["sessions"]
        assert item["session_id"] == "ok" and item["message_count"] == 2  # 工具调用步骤不计入


def test_sessions_empty_when_agent_not_ready():
    with _client(factory=lambda p, s: None) as client:
        response = client.get("/chat/sessions")
        assert response.status_code == 200 and response.json() == {"sessions": []}  # 不是 503


def test_spa_and_assets_served_from_dist():
    dist = _make_dist("dist-ok")
    with _client(ScriptedModel(script=[say("x")]), dist=dist) as client:
        for path in ("/", "/library", "/c/abc-123"):
            response = client.get(path)
            assert response.status_code == 200, (path, response.text)
            assert response.headers["content-type"].startswith("text/html") and response.text == INDEX_HTML
            assert response.headers["cache-control"] == "no-cache"
        asset = client.get("/assets/app-abc123.js")
        assert asset.status_code == 200 and asset.text == "console.log('hi')"
        assert "javascript" in asset.headers["content-type"]
        # 缺失的资源是 404 JSON（不回退到 index.html，否则浏览器会把 HTML 当 JS 解析）；目录穿越被拒绝
        missing = client.get("/assets/nope.js")
        assert missing.status_code == 404 and missing.json()["error"] == "not_found"
        assert client.get("/assets/..%2f..%2fsecret.txt").status_code == 404
        assert client.get("/assets/../secret.txt").status_code == 404
        # 未知路径不由前端兜底
        assert client.get("/no-such-page").status_code == 404
        # 前端路由不接受写方法
        assert client.post("/library").status_code == 405


def test_api_get_routes_not_shadowed_by_frontend():
    dist = _make_dist("dist-api")
    model = ScriptedModel(script=[say("x")])
    with _client(model, dist=dist) as client:
        client.post("/chat", json={"session_id": "keep", "message": "你好"})
        health = client.get("/health")
        assert health.headers["content-type"].startswith("application/json") and health.json()["status"] == "ok"
        docs = client.get("/documents")
        assert docs.headers["content-type"].startswith("application/json")
        assert [d["name"] for d in docs.json()["documents"]] == ["book.pdf", "other.pdf"]
        sessions = client.get("/chat/sessions")
        assert sessions.headers["content-type"].startswith("application/json")
        assert sessions.json()["sessions"][0]["session_id"] == "keep"
        assert client.get("/chat/sessions/keep").json()["session_id"] == "keep"
        job = client.get("/ingest/jobs/nope")
        assert job.status_code == 404 and job.json()["error"] == "not_found"  # JSON 404，不是 index.html
        assert client.get("/openapi.json").json()["info"]["title"] == "RAG 服务"
        # 前端路由不进入 OpenAPI
        paths = client.get("/openapi.json").json()["paths"]
        assert "/library" not in paths and "/" not in paths and "/chat/sessions" in paths


def test_missing_dist_shows_hint_and_api_still_works():
    with _client(ScriptedModel(script=[say("x")]), dist=SCRATCH / "not-built") as client:
        for path in ("/", "/library", "/c/x"):
            response = client.get(path)
            assert response.status_code == 200 and "前端未构建" in response.text and "pnpm build" in response.text
        assert client.get("/assets/app.js").status_code == 404
        assert client.get("/health").status_code == 200
        # dist 在服务运行中被构建出来：立即生效，无需重启
        dist = _make_dist("not-built")
        assert client.get("/").text == INDEX_HTML and client.get("/assets/app-abc123.js").status_code == 200
        assert dist.is_dir()


def test_default_web_dist_setting():
    settings = _settings()
    assert settings.web_dist.is_absolute() and settings.web_dist.parts[-2:] == ("web", "dist")
    relative = _settings(web_dist=Path("custom/out"))
    assert relative.web_dist.is_absolute() and relative.web_dist.parts[-2:] == ("custom", "out")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
