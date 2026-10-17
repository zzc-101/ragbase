"""对话 Agent 的单元测试：假检索流水线 + 脚本化的假工具调用模型（不加载模型、不访问网络、不调用真实大模型）。

覆盖：工具格式与拒答 / 未知书名消息、一轮内多次检索的引用编号稳定、/chat 与 /chat/stream、多轮历史与裁剪、
工具 / 模型调用上限、会话 GET / DELETE、失败回滚与 504 / 502、未配置大模型的 503、会话淘汰、同会话串行。

运行：uv run python tests/test_agent.py
"""
import json
import logging
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import PrivateAttr

import api
from agent_tools import NOT_FOUND_MESSAGE, format_fragment
from chat_agent import MODEL_LIMIT_MESSAGE, SessionRegistry, build_chat_service
from config import Settings
from rag_pipeline import RAGPipeline

logging.disable(logging.CRITICAL)  # 失败路径会 logger.exception，测试输出里不需要这些堆栈

_SCRATCH = tempfile.TemporaryDirectory(prefix="ragdemo-agent-test-")
SCRATCH = Path(_SCRATCH.name)


# ---------------------------------------------------------------- 假检索流水线
class _Retriever:
    """三个候选：c0（book.pdf，第 12 页，有章节）、c1（book.pdf，无页码无章节）、c2（other.pdf）。"""

    NAMES = ["book.pdf", "other.pdf"]

    def __init__(self):
        self.calls = []  # (question, sources)
        self.fail = False
        self.bm25_ready = True

    def catalog(self):
        return [{"name": n, "chunk_count": 10, "heading_chunks": 0, "has_headings": False} for n in self.NAMES]

    def invoke(self, question, mode=None, sources=None):
        self.calls.append((question, sources))
        if self.fail:
            raise RuntimeError("检索内部出错")
        docs = [
            Document(page_content=f"片段文本{i}", metadata={
                "chunk_id": f"c{i}",
                "source": "/data/dir/other.pdf" if i == 2 else "/data/dir/book.pdf",
                "page": 11 if i == 0 else None,
                **({"heading_path": "第1章 概述 > 1.1 范围"} if i == 0 else {}),
            })
            for i in range(3)
        ]
        return [d for d in docs if not sources or Path(d.metadata["source"]).name in sources]


class _Reranker:
    """按问题返回预设的 (候选下标, 得分)；默认 c0=0.99、c1=0.96。"""

    def __init__(self):
        self.by_query = {"红烧肉怎么做": [(0, 0.1)], "B": [(1, 0.99), (2, 0.97)]}

    def rerank_with_scores(self, query, documents, top_k=5):
        scores = self.by_query.get(query, [(0, 0.99), (1, 0.96)])
        return [(i, s) for i, s in scores if i < len(documents)][:top_k]


def _pipeline():
    return RAGPipeline(retriever=_Retriever(), reranker=_Reranker(), answer_chain=None, embeddings=None, device="cpu")


def _settings(**overrides):
    values = dict(
        device="cpu", log_level="WARNING", llm_api_key="sk-test-FAKE-KEY", llm_model="test-model",
        llm_base_url="http://127.0.0.1:1/v1", data_dir=SCRATCH / "data", db_dir=SCRATCH / "chroma",
        meta_file=SCRATCH / "meta.json", cache_dir=SCRATCH / "cache",
    )
    return Settings(_env_file=None, **{**values, **overrides})


# ---------------------------------------------------------------- 脚本化的假模型
_ids = iter(range(1, 10**6))


def search(query, sources=None):
    args = {"query": query, **({"sources": sources} if sources else {})}
    return AIMessage(content="", tool_calls=[{"name": "search_knowledge", "args": args, "id": f"call_{next(_ids)}", "type": "tool_call"}])


def say(text):
    return AIMessage(content=text)


class ScriptedModel(BaseChatModel):
    """按脚本依次回复：脚本项是 AIMessage，或 callable(messages) -> AIMessage / 抛异常；repeat 为真时脚本耗尽后重复最后一项。
    seen 记录每次调用收到的消息。同时实现 _generate 与 _stream（逐 3 个字符流出文本），所以 invoke / stream 两条路都能跑。"""

    script: list[Any]
    repeat: bool = False
    delay: float = 0.0
    _seen: list = PrivateAttr(default_factory=list)
    _cursor: int = PrivateAttr(default=0)
    _guard: Any = PrivateAttr(default_factory=threading.Lock)
    _active: int = PrivateAttr(default=0)
    max_active: int = 0

    @property
    def _llm_type(self):
        return "scripted"

    @property
    def seen(self):
        return self._seen

    def bind_tools(self, tools, **kwargs):
        return self

    def _next(self, messages) -> AIMessage:
        with self._guard:
            self._seen.append(list(messages))
            index = min(self._cursor, len(self.script) - 1) if self.repeat else self._cursor
            self._cursor += 1
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(self.delay)
            step = self.script[index]
            result = step(messages) if callable(step) else step
            return result.model_copy(update={"id": f"run-fake-{index}-{self._cursor}"})
        finally:
            with self._guard:
                self._active -= 1

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        message = self._next(messages)
        if message.tool_calls:
            chunks = [AIMessageChunk(content="", id=message.id, tool_call_chunks=[
                {"name": call["name"], "args": json.dumps(call["args"], ensure_ascii=False), "id": call["id"], "index": i}
                for i, call in enumerate(message.tool_calls)
            ])]
        else:
            text = message.text
            chunks = [AIMessageChunk(content=text[i:i + 3], id=message.id) for i in range(0, len(text), 3)]
        for chunk in chunks:
            yield ChatGenerationChunk(message=chunk)


# ---------------------------------------------------------------- 装配
def _service(model, pipeline=None, **overrides):
    return build_chat_service(pipeline or _pipeline(), _settings(**overrides), model=model)


@contextmanager
def _client(model=None, pipeline=None, factory=None, **overrides):
    """假索引计数 + 假流水线；对话 Agent 默认用传入的假模型构建，factory 可覆盖（如返回 None 或走默认工厂）。"""
    original = api.collection_count
    api.collection_count = lambda *args, **kwargs: 100
    pipeline = pipeline or _pipeline()
    settings = _settings(**overrides)
    kwargs = {} if factory == "default" else {
        "chat_factory": factory or (lambda p, s: build_chat_service(p, s, model=model))
    }
    try:
        app = api.create_app(pipeline_factory=lambda _s: pipeline, settings=settings, **kwargs)
        with TestClient(app) as client:
            client.pipeline = pipeline
            yield client
    finally:
        api.collection_count = original


def _events(response) -> list[tuple[str, dict]]:
    events = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def _assert_error(response, status, error):
    assert response.status_code == status, (response.status_code, response.text)
    body = response.json()
    assert set(body) == {"error", "detail"} and body["error"] == error, body
    return body


def _texts(messages) -> list[str]:
    """模型收到的消息（去掉开头的 system 提示）的文本。"""
    return [m.text for m in messages if m.type != "system"]


def _tool_texts(service, session_id):
    return [m.text for m in service._messages(session_id) if isinstance(m, ToolMessage)]


# ---------------------------------------------------------------- 测试
def test_fragment_format():
    doc = Document(page_content="正文", metadata={"source": "/x/系统架构.pdf", "page": 44, "heading_path": "第3章 > 3.2 节"})
    assert format_fragment(3, doc) == "[3] 来源：《系统架构》> 第3章 > 3.2 节（第45页）\n内容：正文"
    plain = Document(page_content="正文", metadata={"source": "a.md"})
    assert format_fragment(1, plain) == "[1] 来源：《a》\n内容：正文"  # 无章节 / 页码则省略


def test_tool_results_refusal_and_unknown_sources():
    model = ScriptedModel(script=[search("红烧肉怎么做"), say("知识库中没有相关内容")])
    service = _service(model)
    result = service.chat("s1", "红烧肉怎么做")
    assert result.answer == "知识库中没有相关内容" and result.citations == []
    assert _tool_texts(service, "s1") == [NOT_FOUND_MESSAGE]
    assert result.tool_calls == [{"name": "search_knowledge", "args": {"query": "红烧肉怎么做"}}]

    pipeline = _pipeline()
    service = _service(ScriptedModel(script=[search("q", ["nope.pdf"]), say("好")]), pipeline)
    service.chat("s2", "q")
    text = _tool_texts(service, "s2")[0]
    assert "nope.pdf" in text and "book.pdf" in text and "other.pdf" in text  # 说明并列出可选书名
    assert pipeline.retriever.calls == []  # 未知书名在检索前就返回

    service = _service(ScriptedModel(script=[search("q", ["other.pdf"]), say("好")]), pipeline)
    service.chat("s3", "q")
    assert pipeline.retriever.calls == [("q", ["other.pdf"])]
    assert _tool_texts(service, "s3")[0].startswith("[1] 来源：《other》\n内容：片段文本2")


def test_citation_numbers_stable_across_tool_calls():
    # 第一次检索得到 c0 c1 -> [1][2]；第二次得到 c1 c2：c1 仍是 [2]，c2 是新的 [3]
    model = ScriptedModel(script=[search("A"), search("B"), say("结论甲[1]。结论乙[3]。")])
    service = _service(model)
    result = service.chat("s", "问题")
    first, second = _tool_texts(service, "s")
    assert first.startswith("[1] 来源：《book》> 第1章 概述 > 1.1 范围（第12页）\n内容：片段文本0\n\n[2] 来源：《book》\n内容：片段文本1")
    assert second.startswith("[2] 来源：《book》\n内容：片段文本1\n\n[3] 来源：《other》\n内容：片段文本2")
    assert [(n, d.metadata["chunk_id"]) for n, d in result.citations] == [(1, "c0"), (3, "c2")]  # 只返回回答里标注过的，编号不变
    # 下一轮重新编号（登记表按轮新建）
    model.script += [search("A"), say("再答[1]")]
    again = service.chat("s", "再问")
    assert [(n, d.metadata["chunk_id"]) for n, d in again.citations] == [(1, "c0")]


def test_chat_endpoint_happy_path_and_validation():
    model = ScriptedModel(script=[search("挣值分析是什么"), say("挣值分析用于衡量绩效[1]，也可参考[2]。")])
    with _client(model) as client:
        response = client.post("/chat", json={"message": "  挣值分析是什么  "})
        assert response.status_code == 200, response.text
        body = response.json()
        assert set(body) == {"session_id", "answer", "citations", "tool_calls", "took_ms"}
        assert len(body["session_id"]) == 32 and body["answer"].startswith("挣值分析用于衡量绩效[1]")
        assert body["tool_calls"] == [{"name": "search_knowledge", "args": {"query": "挣值分析是什么"}}]
        assert [c["n"] for c in body["citations"]] == [1, 2]
        first = body["citations"][0]
        assert first["chunk_id"] == "c0" and first["source"] == "book.pdf" and first["page"] == 12
        assert first["heading_path"] == "第1章 概述 > 1.1 范围" and first["text"] == "片段文本0"
        assert client.pipeline.retriever.calls[-1][1] is None
        # 校验
        _assert_error(client.post("/chat", json={"message": ""}), 422, "validation_error")
        _assert_error(client.post("/chat", json={"message": "x" * 2001}), 422, "validation_error")
        _assert_error(client.post("/chat", json={"message": "hi", "session_id": "../etc"}), 422, "validation_error")
        _assert_error(client.post("/chat", json={"message": "hi", "session_id": "a" * 65}), 422, "validation_error")
        bad = _assert_error(client.post("/chat", json={"message": "hi", "sources": ["nope.pdf"]}), 400, "bad_request")
        assert "nope.pdf" in bad["detail"]
        assert client.get("/health").json()["agent_ready"] is True


def test_request_sources_used_as_default_filter():
    model = ScriptedModel(script=[search("q"), search("q", ["book.pdf"]), say("ok")])
    with _client(model) as client:
        assert client.post("/chat", json={"message": "m", "sources": ["other.pdf"]}).status_code == 200
        # 模型没传 sources -> 用请求里的默认过滤；模型显式传了 -> 以模型为准
        assert client.pipeline.retriever.calls == [("q", ["other.pdf"]), ("q", ["book.pdf"])]


def test_multi_turn_history_and_isolation():
    model = ScriptedModel(script=[say("你好，我是助手"), say("你刚才问候了我"), say("新会话")])
    service = _service(model)
    service.chat("a", "你好")
    service.chat("a", "我刚才说了什么？")
    assert _texts(model.seen[1]) == ["你好", "你好，我是助手", "我刚才说了什么？"]  # 第二轮能看到第一轮
    assert model.seen[1][0].type == "system" and "search_knowledge" in model.seen[1][0].text
    service.chat("b", "你好吗")
    assert _texts(model.seen[2]) == ["你好吗"]  # 别的会话互不可见


def test_history_is_bounded_by_turns():
    model = ScriptedModel(script=[say(f"答{i}") for i in range(4)])
    service = _service(model, chat_history_turns=2)
    for i in range(4):
        service.chat("s", f"问{i}")
    assert _texts(model.seen[3]) == ["问2", "答2", "问3"]  # 保留最近 2 轮（含当前），整轮丢弃
    assert [m["content"] for m in service.history("s")] == ["问2", "答2", "问3", "答3"]


def test_tool_call_limit_is_graceful():
    # 上限 1 次：第二次检索被拒绝（检索器只被调用一次），模型据已有结果作答
    model = ScriptedModel(script=[search("A"), search("B"), say("据已有片段回答[1]")])
    pipeline = _pipeline()
    service = _service(model, pipeline, agent_max_tool_calls=1)
    result = service.chat("s", "问")
    assert len(pipeline.retriever.calls) == 1
    assert result.answer == "据已有片段回答[1]" and [n for n, _ in result.citations] == [1]
    assert "limit" in _tool_texts(service, "s")[1].lower()  # 第二个 ToolMessage 是上限提示


def test_model_call_limit_ends_with_chinese_message():
    model = ScriptedModel(script=[search("A")], repeat=True)  # 模型一直要检索
    service = _service(model, agent_max_model_calls=2)
    result = service.chat("s", "问")
    assert result.answer == MODEL_LIMIT_MESSAGE and len(model.seen) == 2
    # 流式：中间件直接给出的提示也会作为 token 发出
    model = ScriptedModel(script=[search("A")], repeat=True)
    service = _service(model, agent_max_model_calls=2)
    events = list(service.stream("s", "问"))
    assert [e for e, _ in events if e == "token"] == ["token"] and events[-1] == ("done", {"answer": MODEL_LIMIT_MESSAGE})


def test_chat_stream_event_sequence():
    model = ScriptedModel(script=[search("挣值"), say("答案甲[1]，答案乙[2]。")])
    with _client(model) as client:
        response = client.post("/chat/stream", json={"message": "挣值", "session_id": "sid-1"})
        assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
        events = _events(response)
    names = [name for name, _ in events]
    assert names[0] == "session" and events[0][1] == {"session_id": "sid-1"}
    assert names[1] == "tool_call" and events[1][1] == {"name": "search_knowledge", "args": {"query": "挣值"}}
    assert set(names[2:-2]) == {"token"} and len(names) > 5  # 文本被切成多个 token
    assert names[-2:] == ["citations", "done"]
    assert "".join(data["text"] for name, data in events if name == "token") == "答案甲[1]，答案乙[2]。"
    citations = events[-2][1]["citations"]
    assert [c["n"] for c in citations] == [1, 2] and citations[0]["source"] == "book.pdf"
    done = events[-1][1]
    assert done["answer"] == "答案甲[1]，答案乙[2]。" and isinstance(done["took_ms"], float)


def test_chat_stream_error_event_and_pre_stream_errors():
    model = ScriptedModel(script=[lambda messages: (_ for _ in ()).throw(httpx.ReadTimeout("slow"))])
    with _client(model) as client:
        events = _events(client.post("/chat/stream", json={"message": "q", "session_id": "e1"}))
        assert [name for name, _ in events] == ["session", "error"]
        assert events[1][1]["error"] == "llm_timeout" and set(events[1][1]) == {"error", "detail"}
        _assert_error(client.post("/chat/stream", json={"message": ""}), 422, "validation_error")
        _assert_error(client.post("/chat/stream", json={"message": "q", "sources": ["x"]}), 400, "bad_request")


def test_session_get_and_delete():
    model = ScriptedModel(script=[search("A"), say("答[1]")])
    with _client(model) as client:
        client.post("/chat", json={"message": "问", "session_id": "keep-1"})
        body = client.get("/chat/sessions/keep-1").json()
        assert body["session_id"] == "keep-1"
        roles = [m["role"] for m in body["messages"]]
        assert roles == ["user", "assistant", "tool", "assistant"], body
        assert body["messages"][1]["tool_calls"] == [{"name": "search_knowledge", "args": {"query": "A"}}]
        assert body["messages"][2]["content"].startswith("[search_knowledge] [1] 来源：《book》") and len(body["messages"][2]["content"]) < 120
        assert body["messages"][3] == {"role": "assistant", "content": "答[1]", "tool_calls": None}
        _assert_error(client.get("/chat/sessions/nope"), 404, "not_found")
        _assert_error(client.get("/chat/sessions/bad%20id"), 422, "validation_error")
        assert client.delete("/chat/sessions/keep-1").json() == {"session_id": "keep-1", "deleted": True}
        _assert_error(client.get("/chat/sessions/keep-1"), 404, "not_found")
        _assert_error(client.delete("/chat/sessions/keep-1"), 404, "not_found")


def test_llm_errors_map_to_504_502_and_failed_turn_is_rolled_back():
    def boom_after_tool(messages):
        raise RuntimeError("upstream exploded sk-secret")

    # 检索之后大模型失败：本轮（含 tool_calls）整体回滚，下一轮不会带着缺 ToolMessage 的历史
    model = ScriptedModel(script=[search("A"), boom_after_tool, say("恢复了")])
    with _client(model) as client:
        body = _assert_error(client.post("/chat", json={"message": "问", "session_id": "r1"}), 502, "llm_error")
        assert "sk-secret" not in json.dumps(body, ensure_ascii=False)
        _assert_error(client.get("/chat/sessions/r1"), 404, "not_found")
        ok = client.post("/chat", json={"message": "再问", "session_id": "r1"})
        assert ok.status_code == 200 and ok.json()["answer"] == "恢复了"
        assert [m["content"] for m in client.get("/chat/sessions/r1").json()["messages"]] == ["再问", "恢复了"]

    timeout = ScriptedModel(script=[lambda messages: (_ for _ in ()).throw(httpx.ReadTimeout("slow"))])
    with _client(timeout) as client:
        _assert_error(client.post("/chat", json={"message": "问"}), 504, "llm_timeout")


def test_retrieval_failure_is_internal_error_not_llm_error():
    pipeline = _pipeline()
    pipeline.retriever.fail = True
    model = ScriptedModel(script=[search("A"), say("x")])
    with _client(model, pipeline) as client:
        response = client.post("/chat", json={"message": "问"})
        _assert_error(response, 500, "internal_error")


def test_chat_503_when_llm_not_configured():
    unconfigured = dict(llm_api_key=None, llm_model=None, llm_base_url=None)
    for kwargs in ({"factory": lambda p, s: None}, {"factory": "default", **unconfigured}):
        with _client(**kwargs) as client:
            assert client.get("/health").json()["agent_ready"] is False
            for response in (
                client.post("/chat", json={"message": "你好"}),
                client.post("/chat/stream", json={"message": "你好"}),
                client.get("/chat/sessions/x"),
                client.delete("/chat/sessions/x"),
            ):
                body = _assert_error(response, 503, "service_unavailable")
                assert "大模型未配置" in body["detail"]


def test_session_registry_ttl_lru_and_busy_sessions():
    now, evicted = [0.0], []
    registry = SessionRegistry(max_sessions=2, ttl=100, on_evict=evicted.append, clock=lambda: now[0])
    for name in ("a", "b"):
        with registry.session(name):
            now[0] += 1
    with registry.session("c"):  # 超出上限：最久未用的 a 被淘汰
        assert evicted == ["a"] and not registry.known("a") and registry.known("b")
    now[0] += 1000  # 全部过期
    assert not registry.known("b") and not registry.known("c") and sorted(evicted) == ["a", "b", "c"]
    assert len(registry) == 0
    # 忙的会话不被淘汰，即使超过 TTL / 上限
    with registry.session("x"):
        now[0] += 1000
        with registry.session("y"):
            pass
        assert registry.known("x")


def test_evicted_session_history_is_deleted():
    service = _service(ScriptedModel(script=[say("一"), say("二")]), chat_max_sessions=1)
    service.chat("s1", "问")
    service.chat("s2", "问")  # 上限 1：s1 被淘汰，同时从 checkpointer 删除
    assert service.history("s1") is None and service._messages("s1") == [] and len(service.history("s2")) == 2


def test_same_session_requests_are_serialized():
    model = ScriptedModel(script=[say("一"), say("二"), say("三")], delay=0.05)
    service = _service(model)
    threads = [threading.Thread(target=service.chat, args=("same", f"问{i}")) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert model.max_active == 1 and len(service.history("same")) == 6  # 三轮都完整落盘，没有互相覆盖
    other = ScriptedModel(script=[say("x")] * 2, delay=0.1)
    service = _service(other)
    threads = [threading.Thread(target=service.chat, args=(f"s{i}", "问")) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert other.max_active == 2  # 不同会话可以并行


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
