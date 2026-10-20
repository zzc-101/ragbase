"""对话会话持久化的测试：文件库 + 假检索流水线 + 脚本化的假工具调用模型（不加载模型、不访问网络、不调用真实大模型）。

覆盖：重启后会话列表 / 历史 / 引用编号保持一致且编号接着走、历史窗口（状态完整、模型只看到最近 N 轮，同步与流式）、标题固定与累计消息数、
失败回滚落库（无残留）、进程中途被杀后的半截轮次恢复、无 chunk_id 片段的跨进程稳定引用 key、删除（含等当前轮结束、重启后不复活）、
checkpoint 瘦身、对话 Agent 未就绪时会话接口读库、废弃配置项的警告、多会话并发写库。

所有数据库都在临时目录里；不会在项目根目录生成 ragbase.db。

运行：uv run python tests/test_chat_persistence.py
"""
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ragbase.agent.service import build_chat_service, incomplete_turn_start
from ragbase.agent.store import ChatStore
from ragbase.agent.tools import CitationRegistry
from ragbase.config import Settings
from ragbase.db import open_database
from ragbase.retrieval.pipeline import GenerationError

# test_agent 提供假流水线 / 脚本化假模型 / 临时目录（它还会 logging.disable，需要看日志的测试自己恢复）
from test_agent import (  # noqa: E402  (tests 目录在 sys.path 里)
    ScriptedModel, _assert_error, _client, _pipeline, _settings, say, search,
)

_SCRATCH = tempfile.TemporaryDirectory(prefix="ragdemo-chat-persist-test-")
SCRATCH = Path(_SCRATCH.name)
_dirs = iter(range(1, 10**6))


def _fresh_db_path(name: str = "ragbase.db") -> Path:
    directory = SCRATCH / f"case-{next(_dirs)}"
    directory.mkdir()
    return directory / name


@contextmanager
def _running(db_path: Path, model, pipeline=None, **overrides):
    """像一次“服务启动”：打开（或创建）库文件、构造 ChatService；退出时像关机一样关掉它们。产出 (service, db)。"""
    settings = _settings(db_path=db_path, **overrides)
    db = open_database(settings)
    service = build_chat_service(pipeline or _pipeline(), settings, model=model, db=db)
    try:
        yield service, db
    finally:
        service.close()
        db.close()


def _cited(citations):
    return [(n, d.metadata["chunk_id"]) for n, d in citations]


def _count(db, sql, params=()):
    return db.query(sql, params)[0][0]


def _normalized(history):
    """历史接口的返回 -> 可比较的纯数据（Document 展开成 (正文, 元数据)）。"""
    return [
        {**m, "citations": [(n, d.page_content, d.metadata) for n, d in m["citations"]]} if "citations" in m else m
        for m in history
    ]


# ---------------------------------------------------------------- 测试
def test_restart_keeps_sessions_history_and_citations():
    path = _fresh_db_path()
    # 第 1 轮 A -> c0 c1 = [1][2]；第 2 轮 B -> c1 c2：c2 是 [3]；第 3 轮不调用工具，沿用 [1]
    first_run = ScriptedModel(script=[
        search("A"), say("甲[1]乙[2]"),
        search("B"), say("丙[3]，乙[2]"),
        say("沿用[1]"),
        say("另一个会话"),
    ])
    with _running(path, first_run) as (service, _):
        for message in ("问1", "问2", "问3"):
            service.chat("s1", message)
        service.chat("s2", "闲聊")
        sessions_before = service.list_sessions()
        history_before = _normalized(service.history("s1"))
        assert [s["session_id"] for s in sessions_before] == ["s2", "s1"]
        assert [m["role"] for m in history_before] == ["user", "assistant", "tool", "assistant", "user", "assistant", "tool", "assistant", "user", "assistant"]

    # “重启”：同一个库文件，全新的 ChatService / 模型 / 引用登记表
    second_run = ScriptedModel(script=[search("A"), search("C"), say("旧[1]，新[4][5]，沿用[2]")])
    with _running(path, second_run) as (service, _):
        assert service.list_sessions() == sessions_before
        assert _normalized(service.history("s1")) == history_before
        assert _cited(service.history("s1")[3]["citations"]) == [(1, "c0"), (2, "c1")]
        # 第 4 轮：A 的 c0 仍是 [1]；C 得到 c3 c4 c1：新片段接着之前的最大编号（3）编成 [4][5]，c1 仍是 [2]
        result = service.chat("s1", "问4")
        assert _cited(result.citations) == [(1, "c0"), (2, "c1"), (4, "c3"), (5, "c4")]
        tool_texts = [m.text for m in service._messages("s1") if isinstance(m, ToolMessage)]
        assert tool_texts[-1].startswith("[4] ") and "[5] " in tool_texts[-1] and tool_texts[-1].rstrip().endswith("片段文本1")
        assert tool_texts[-1].count("[2] 来源") == 1
        assert len(service.history("s1")) == len(history_before) + 6  # 用户消息 + 两次检索（各一条 tool_calls 和一条 ToolMessage）+ 最终回答


def test_full_history_kept_but_model_sees_window():
    path = _fresh_db_path()
    model = ScriptedModel(script=[say(f"答{i}") for i in range(5)])
    with _running(path, model, chat_history_turns=2) as (service, _):
        for i in range(3):
            service.chat("s", f"问{i}")
        service.chat("s", "问3")
        users = [m for m in service.history("s") if m["role"] == "user"]
        assert len(users) == 4  # 库里 / 历史接口里是全部消息
        sent = [m for m in model.seen[3] if m.type == "human"]
        assert [m.text for m in sent] == ["问2", "问3"]  # 模型第 4 轮只收到最近 2 条用户消息
        assert [m.text for m in model.seen[3] if m.type == "ai"] == ["答2"]
    # 重启后窗口规则不变：历史完整，第 5 轮模型仍只看到最近 2 轮；流式路径同样
    model = ScriptedModel(script=[say("答4")])
    with _running(path, model, chat_history_turns=2) as (service, _):
        events = list(service.stream("s", "问4"))
        assert events[-1] == ("done", {"answer": "答4"})
        assert [m.text for m in model.seen[0] if m.type == "human"] == ["问3", "问4"]
        assert len([m for m in service.history("s") if m["role"] == "user"]) == 5


def test_title_fixed_and_count_cumulative():
    path = _fresh_db_path()
    model = ScriptedModel(script=[search("A"), say("甲[1]"), say("二"), say("三")])
    with _running(path, model) as (service, db):
        now = [1000.0]
        service.clock = lambda: now[0]
        service.chat("s", "  第一个问题\n分成两行，而且非常非常非常非常非常非常非常非常长，肯定超过三十个字  ")
        first = service.list_sessions()[0]
        assert "\n" not in first["title"] and first["title"].endswith("…") and first["title"].startswith("第一个问题 分成两行")
        assert first["message_count"] == 2 and first["updated_at"] == 1000.0  # 用户消息 + 最终回答；工具步骤不计
        now[0] = 2000.0
        service.chat("s", "第二个问题")
        service.chat("s", "第三个问题")
        item = service.list_sessions()[0]
        assert item["title"] == first["title"]  # 标题固定为第一条用户消息
        assert item["message_count"] == 6 and item["updated_at"] == 2000.0
        assert _count(db, "SELECT created_at FROM chat_sessions") == 1000.0
        # 与历史接口口径一致：用户消息 + 不带 tool_calls 的助手消息
        history = service.history("s")
        assert item["message_count"] == sum(1 for m in history if m["role"] == "user" or (m["role"] == "assistant" and "tool_calls" not in m))


def test_failed_turn_rollback_persists():
    def boom(messages):
        raise RuntimeError("upstream exploded")

    path = _fresh_db_path()
    # 会话 f：首轮检索后大模型失败 -> 库里什么都不剩；会话 g：第 2 轮检索 c3 c4 后失败 -> 回到第 1 轮之后
    model = ScriptedModel(script=[
        search("A"), boom,
        search("A"), say("甲[1]乙[2]"),
        search("C"), boom,
    ])
    with _running(path, model) as (service, db):
        try:
            service.chat("f", "问")
            raise AssertionError("应当抛出 GenerationError")
        except GenerationError:
            pass
        assert service.list_sessions() == [] and service.history("f") is None
        assert _count(db, "SELECT COUNT(*) FROM chat_sessions") == 0
        assert _count(db, "SELECT COUNT(*) FROM chat_citations") == 0  # 首轮登记的 [1][2] 也撤销了
        assert _count(db, "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'f'") == 0

        service.chat("g", "问1")
        before = service.list_sessions()[0]
        history_before = _normalized(service.history("g"))
        try:
            service.chat("g", "问2")
            raise AssertionError("应当抛出 GenerationError")
        except GenerationError:
            pass
        assert [row["number"] for row in db.query("SELECT number FROM chat_citations WHERE session_id = 'g' ORDER BY number")] == [1, 2]
        assert service.list_sessions() == [before]  # message_count / updated_at 都没变
        assert _normalized(service.history("g")) == history_before
        assert _count(db, "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'g'") == 1
    # 重启后同样干净
    with _running(path, ScriptedModel(script=[search("C"), say("新[3][4]")])) as (service, _):
        result = service.chat("g", "问2")
        assert _cited(result.citations) == [(3, "c3"), (4, "c4")]  # 编号接着失败前的 2 往下编


def test_recover_incomplete_turn_after_crash():
    path = _fresh_db_path()
    ghost = Document(page_content="多登记的片段", metadata={"chunk_id": "ghost"})
    with _running(path, ScriptedModel(script=[search("A"), say("甲[1]乙[2]")])) as (service, db):
        service.chat("s", "问1")
        assert service.store.committed_citations("s") == 2
        # 模拟进程在第 2 轮中途被杀：状态里留下“用户消息 + 带 tool_calls 的 AI 消息”，工具结果缺失；引用表多出本轮登记的第 3 行
        service.agent.update_state(service._config("s"), {"messages": [HumanMessage(content="半截的问题"), search("B")]})
        service.store.add_citation("s", 3, "id:ghost", ghost)
        assert incomplete_turn_start(service._messages("s")) == 4
        assert [n for n, _, _ in service.store.load_citations("s")] == [1, 2, 3]
        count_before = service.list_sessions()[0]["message_count"]

    # 重启后的下一轮：残缺的一轮和多余的引用都被清掉，再正常完成
    with _running(path, ScriptedModel(script=[search("C"), say("新[3]"), say("再答")])) as (service, db):
        result = service.chat("s", "问2")
        assert result.answer == "新[3]" and _cited(result.citations) == [(3, "c3")]  # [3] 空出来了，被本轮的真实片段占用
        texts = [m["content"] for m in service.history("s") if m["role"] in ("user", "assistant") and not m.get("tool_calls")]
        assert texts == ["问1", "甲[1]乙[2]", "问2", "新[3]"]
        assert "半截的问题" not in str([m["content"] for m in service.history("s")])
        keys = [row["chunk_key"] for row in db.query("SELECT chunk_key FROM chat_citations WHERE session_id = 's' ORDER BY number")]
        assert keys == ["id:c0", "id:c1", "id:c3", "id:c4"]  # 多登记的 ghost 行没有了，[3][4] 是本轮的真实片段
        assert service.list_sessions()[0]["message_count"] == count_before + 2  # 残缺的那轮不计数
        # 工具结果只对上一部分也算残缺
        two_calls = AIMessage(content="", tool_calls=[
            {"name": "search_knowledge", "args": {"query": "x"}, "id": "t1", "type": "tool_call"},
            {"name": "list_documents", "args": {}, "id": "t2", "type": "tool_call"},
        ])
        answered = ToolMessage(content="r", tool_call_id="t1", name="search_knowledge")
        assert incomplete_turn_start([HumanMessage(content="q"), two_calls, answered]) == 0
        assert incomplete_turn_start([HumanMessage(content="q"), two_calls, answered, ToolMessage(content="r", tool_call_id="t2", name="list_documents"), AIMessage(content="好")]) is None
        assert incomplete_turn_start([HumanMessage(content="q")]) == 0 and incomplete_turn_start([]) is None

    # 第 1 轮就中途被杀：会话行不存在，整个 thread 清空，引用全部清掉；之后可以正常对话
    path = _fresh_db_path()
    with _running(path, ScriptedModel(script=[say("好"), say("好")])) as (service, db):
        service.agent.update_state(service._config("n"), {"messages": [HumanMessage(content="半截"), search("A")]})
        service.store.add_citation("n", 1, "id:ghost", ghost)
        assert service.list_sessions() == []
    with _running(path, ScriptedModel(script=[say("全新的回答")])) as (service, db):
        service.chat("n", "重新开始")
        assert [m["content"] for m in service.history("n")] == ["重新开始", "全新的回答"]
        assert _count(db, "SELECT COUNT(*) FROM chat_citations") == 0
        assert service.store.committed_citations("n") == 0


def test_citation_key_stable_without_chunk_id():
    doc = Document(page_content="没有 chunk_id 的正文", metadata={"source": "/data/a.pdf", "page": 3})
    key = CitationRegistry._key(doc)
    code = (
        "from langchain_core.documents import Document\n"
        "from ragbase.agent.tools import CitationRegistry\n"
        "print(CitationRegistry._key(Document(page_content='没有 chunk_id 的正文', metadata={'source': '/data/a.pdf', 'page': 3})))\n"
    )
    for seed in ("1", "2", "random"):  # 内置 hash() 受 PYTHONHASHSEED 影响，每个进程都不同；稳定哈希不受影响
        output = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, env={**os.environ, "PYTHONHASHSEED": seed}
        ).stdout.strip()
        assert output == key, (seed, output, key)
    assert key.startswith("text:") and "id:" not in key
    assert CitationRegistry._key(Document(page_content="别的正文", metadata={"source": "/data/a.pdf", "page": 3})) != key
    assert CitationRegistry._key(Document(page_content="没有 chunk_id 的正文", metadata={"source": "/data/a.pdf", "page": 4})) != key
    assert CitationRegistry._key(Document(page_content="x", metadata={"chunk_id": "c9"})) == "id:c9"
    # 落库后重新打开：同一个片段还是原来的编号
    db = open_database(_settings(db_path=_fresh_db_path()))
    try:
        store = ChatStore(db)
        assert CitationRegistry(store, "s").add(doc) == 1
        reopened = CitationRegistry(store, "s")
        assert reopened.add(Document(page_content="另一个", metadata={})) == 2
        assert reopened.add(doc) == 1 and CitationRegistry(store, "s").mark() == 2
    finally:
        db.close()


def test_delete_removes_everything_and_survives_restart():
    path = _fresh_db_path()
    model = ScriptedModel(script=[search("A"), say("甲[1]"), search("B"), say("乙[1]")])
    with _running(path, model) as (service, db):
        service.chat("keep", "问")
        service.chat("gone", "问")
        assert service.delete("gone") is True
        assert service.delete("gone") is False and service.history("gone") is None
        assert [s["session_id"] for s in service.list_sessions()] == ["keep"]
    with _running(path, ScriptedModel(script=[say("x")])) as (service, db):
        assert [s["session_id"] for s in service.list_sessions()] == ["keep"]
        assert service.history("gone") is None and service._messages("gone") == []
        for table in ("checkpoints", "writes"):
            assert _count(db, f"SELECT COUNT(*) FROM {table} WHERE thread_id = 'gone'") == 0, table
        assert _count(db, "SELECT COUNT(*) FROM chat_citations WHERE session_id = 'gone'") == 0
        assert _count(db, "SELECT COUNT(*) FROM chat_sessions WHERE session_id = 'gone'") == 0
        assert _count(db, "SELECT COUNT(*) FROM chat_citations WHERE session_id = 'keep'") > 0  # 别的会话不受影响

    # delete 要等当前这一轮结束才执行：不会被这一轮的提交“复活”
    slow = ScriptedModel(script=[say("慢慢答")], delay=0.3)
    with _running(_fresh_db_path(), slow) as (service, db):
        worker = threading.Thread(target=service.chat, args=("busy", "问"))
        worker.start()
        time.sleep(0.1)
        assert service.delete("busy") is True  # 阻塞到那一轮结束后才删，且这时会话已经存在
        worker.join()
        assert service.list_sessions() == [] and service.history("busy") is None
        assert _count(db, "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'busy'") == 0


def test_old_checkpoints_pruned():
    path = _fresh_db_path()
    model = ScriptedModel(script=[search("A"), say("甲[1]"), search("B"), say("乙[1]"), say("丙"), say("别人")])
    with _running(path, model) as (service, db):
        for i in range(3):
            service.chat("s", f"问{i}")
            assert _count(db, "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 's'") == 1
        latest = db.query("SELECT checkpoint_id FROM checkpoints WHERE thread_id = 's'")[0]["checkpoint_id"]
        assert {row[0] for row in db.query("SELECT DISTINCT checkpoint_id FROM writes WHERE thread_id = 's'")} <= {latest}
        assert len(service.history("s")) == 3 * 2 + 4  # 瘦身不丢消息：3 条用户 + 3 条回答 + 2 轮检索的 tool_calls / ToolMessage
        # 别的会话的 checkpoint 不受影响
        service.chat("other", "问")
    with _running(path, ScriptedModel(script=[say("丁")])) as (service, db):
        service.chat("s", "问3")
        assert _count(db, "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 's'") == 1
        assert _count(db, "SELECT COUNT(*) FROM checkpoints WHERE thread_id = 'other'") == 1
        assert len([m for m in service.history("s") if m["role"] == "user"]) == 4
    # 内存模式（db=None）：不瘦身也能正常工作
    memory = build_chat_service(_pipeline(), _settings(), model=ScriptedModel(script=[say("一"), say("二")]))
    memory.chat("m", "问")
    memory.chat("m", "再问")
    assert len(memory.history("m")) == 4 and len(memory.list_sessions()) == 1


def test_sessions_endpoints_when_agent_not_ready():
    path = _fresh_db_path()
    model = ScriptedModel(script=[search("A"), say("答[1]与[2]")])
    with _running(path, model) as (service, _):
        service.chat("saved-1", "库里的问题")
    # 对话 Agent 没有就绪（工厂返回 None）：会话读 / 删直接读同一个库文件
    with _client(factory=lambda p, s: None, db_path=path) as client:
        assert client.get("/health").json()["agent_ready"] is False
        sessions = client.get("/chat/sessions").json()["sessions"]
        assert [s["session_id"] for s in sessions] == ["saved-1"]
        assert sessions[0]["title"] == "库里的问题" and sessions[0]["message_count"] == 2
        body = client.get("/chat/sessions/saved-1").json()
        assert [m["role"] for m in body["messages"]] == ["user", "assistant", "tool", "assistant"]
        last = body["messages"][-1]
        assert last["content"] == "答[1]与[2]" and [c["n"] for c in last["citations"]] == [1, 2] and last["citations"][0]["chunk_id"] == "c0"
        _assert_error(client.get("/chat/sessions/nope"), 404, "not_found")
        _assert_error(client.post("/chat", json={"message": "你好"}), 503, "service_unavailable")
        _assert_error(client.post("/chat/stream", json={"message": "你好"}), 503, "service_unavailable")
        assert client.delete("/chat/sessions/saved-1").json() == {"session_id": "saved-1", "deleted": True}
        _assert_error(client.get("/chat/sessions/saved-1"), 404, "not_found")
        _assert_error(client.delete("/chat/sessions/saved-1"), 404, "not_found")
        assert client.get("/chat/sessions").json() == {"sessions": []}
    db = open_database(_settings(db_path=path))
    try:  # 删得干净：checkpoint / 引用 / 会话行都没了
        assert _count(db, "SELECT COUNT(*) FROM checkpoints") == 0 and _count(db, "SELECT COUNT(*) FROM chat_citations") == 0
    finally:
        db.close()
    # 应用启动时打开的库就是配置里的那个文件；服务就绪时同一个库里的会话在重启后仍可见
    model = ScriptedModel(script=[say("你好呀")])
    with _client(model, db_path=path) as client:
        session_id = client.post("/chat", json={"message": "你好"}).json()["session_id"]
    with _client(factory=lambda p, s: None, db_path=path) as client:
        assert [s["session_id"] for s in client.get("/chat/sessions").json()["sessions"]] == [session_id]


def test_deprecated_session_settings_warn():
    class Capture(logging.Handler):
        def __init__(self):
            super().__init__(logging.WARNING)
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    capture = Capture()
    logger = logging.getLogger("ragbase.agent.service")
    logger.addHandler(capture)
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    logging.disable(logging.NOTSET)  # test_agent 为了安静把日志全关了，这里需要看到 warning
    original = {key: os.environ.get(key) for key in ("RAG_CHAT_SESSION_TTL", "RAG_CHAT_MAX_SESSIONS")}
    try:
        for key in original:
            os.environ.pop(key, None)
        build_chat_service(_pipeline(), Settings(_env_file=None), model=ScriptedModel(script=[say("x")]))
        assert not [m for m in capture.messages if "已废弃" in m]  # 没设置就不提示

        os.environ["RAG_CHAT_SESSION_TTL"] = "120"
        settings = Settings(_env_file=None)
        assert settings.chat_session_ttl == 120 and "chat_session_ttl" in settings.model_fields_set  # 仍能解析
        service = build_chat_service(_pipeline(), settings, model=ScriptedModel(script=[say("照常工作")]))
        assert service.chat("s", "问").answer == "照常工作"
        warned = [m for m in capture.messages if "已废弃" in m]
        assert len(warned) == 1 and "RAG_CHAT_SESSION_TTL" in warned[0] and "RAG_CHAT_MAX_SESSIONS" not in warned[0], warned

        # 经 create_app 的启动流程也能启动，同样有警告
        capture.messages.clear()
        os.environ["RAG_CHAT_MAX_SESSIONS"] = "5"
        with _client(ScriptedModel(script=[say("你好")])) as client:
            assert client.post("/chat", json={"message": "hi"}).status_code == 200
        warned = [m for m in capture.messages if "已废弃" in m]
        assert len(warned) == 1 and "RAG_CHAT_SESSION_TTL" in warned[0] and "RAG_CHAT_MAX_SESSIONS" in warned[0], warned
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        logging.disable(logging.CRITICAL)
        logger.removeHandler(capture)
        logger.setLevel(previous_level)


def test_concurrent_sessions_write_without_locked_errors():
    path = _fresh_db_path()
    # 单个脚本项 + repeat：按收到的消息决定——最后一条是工具结果就作答，否则先检索
    model = ScriptedModel(script=[lambda messages: say("答[1]") if messages[-1].type == "tool" else search("A")], repeat=True, delay=0.02)
    errors = []
    with _running(path, model) as (service, db):
        def work(index):
            try:
                for turn in range(3):
                    service.chat(f"c{index}", f"问{turn}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors, errors
        sessions = service.list_sessions()
        assert len(sessions) == 8 and all(s["message_count"] == 6 for s in sessions)
        assert _count(db, "SELECT COUNT(*) FROM checkpoints") == 8  # 每个会话只剩最新的一个 checkpoint


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
