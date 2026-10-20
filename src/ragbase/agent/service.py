"""对话 Agent：LangChain create_agent + 知识库工具 + 按会话的多轮记忆（持久化到内置 SQLite）。

- 一次启动构建一次（build_chat_service）；会话 = LangGraph 的 thread，thread_id 就是 session_id。
  传入 Database 时历史由 SqliteSaver 保存、会话摘要和引用编号存在 chat_sessions / chat_citations（ragbase/agent/store.py），重启后都还在；
  不传（命令行 chat、部分测试）则用 InMemorySaver + 内存库，只活在进程里。两种模式走同一套代码。
- 状态里保留全部消息，只在调用大模型时截取最近 chat_history_turns 轮（HistoryTrimMiddleware 的 wrap_model_call 钩子，不回写状态；
  一轮 = 一条用户消息及其后的全部回复 / 工具消息，整轮保留或丢弃，所以不会留下缺 ToolMessage 的 tool_calls）。
  不用 SummarizationMiddleware，避免额外的大模型调用。
- 会话不自动淘汰（没有 TTL / 数量上限），只有用户删除（DELETE /chat/sessions/{id}）才会清掉；每轮成功后只保留该会话最新的一个 checkpoint，库不会随轮数膨胀。
- 同一会话的请求串行（会话锁，只存在内存里）；失败 / 客户端断开的那一轮会回滚（消息和引用编号都恢复到本轮之前），不会留下半截状态
  （否则下一轮的历史里可能有缺 ToolMessage 的 tool_calls，OpenAI 兼容服务会拒绝）。
  进程在一轮中途被杀（kill -9 / 断电）时来不及回滚：下一轮开始时检测状态末尾不完整的一轮，整轮裁掉，并把引用表截回“已提交编号”。
- 引用编号按会话保存（CitationRegistry 写穿到 chat_citations）：跨轮连续、同一片段编号不变（含重启后）；失败回滚时一并撤销本轮新登记的片段；
  会话删除时随之清除；历史窗口截取不影响编号，被截出模型上下文的旧轮次在历史接口里仍带引用。
- 限制：每轮工具调用上限（超出后工具被拒绝，模型据已有结果作答）、每轮模型调用上限（超出后直接结束并给出提示）。
"""
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware, hook_config
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, RemoveMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from ragbase.agent.store import TITLE_MAX_CHARS, ChatStore  # noqa: F401  TITLE_MAX_CHARS 是对外常量，保持从本模块可导入
from ragbase.agent.tools import ChatContext, CitationRegistry, ToolRetrievalError, build_tools
from ragbase.config import DEPRECATED_SETTINGS, Settings
from ragbase.db import Database
from ragbase.llm.factory import create_chat_model
from ragbase.retrieval.pipeline import GenerationError, RAGPipeline, is_timeout_error

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是知识库问答助手，知识库收录了《系统架构设计师》《信息系统项目管理师》等教材。

规则：
1. 用户的问题涉及知识库里的专业内容（概念、方法、公式、流程、考点等）时，必须先调用 search_knowledge 检索，再回答。
2. 调用 search_knowledge 时，query 必须是自包含的完整问题：结合对话历史补全代词和省略，不要直接照抄“它”“那个”“上面说的”这类说法。
3. 只能依据工具返回的片段回答，不要补充片段里没有的事实、数据或常识；片段只覆盖问题的一部分时，只回答有依据的部分，并说明其余部分知识库里没有。
4. 在引用了某个片段的句子末尾用 [编号] 标注，编号取自工具返回的 [n]。编号在整个对话内都有效：可以沿用本对话里更早的工具结果中出现过的编号；不要自己编造任何工具结果里没出现过的编号。
5. 工具表示没有找到可靠的相关内容时，回答“知识库中没有相关内容”，不要编造，也不要换着说法反复检索。
6. 寒暄、问候、关于你自己的问题可以直接回答，不需要调用工具。
7. 回答简洁，直接给结论和依据。"""

MODEL_LIMIT_MESSAGE = "本轮处理步骤过多，已停止。请把问题拆小一些，或换个问法再试。"
EMPTY_ANSWER_MESSAGE = "（模型没有给出回答，请重试）"


class HistoryTrimMiddleware(AgentMiddleware):
    """调用大模型前把消息截到最近 max_turns 轮；只改发给模型的请求，不改（也不写回）状态，所以库里保存全部消息。

    整轮保留或丢弃（从某条用户消息开始切），不会拆开 tool_calls 与 ToolMessage。同步 / 异步 / 流式调用走的是同一个模型节点，所以都生效。
    """

    def __init__(self, max_turns: int):
        super().__init__()
        self.max_turns = max_turns

    def window(self, messages: list) -> list:
        starts = [index for index, message in enumerate(messages) if isinstance(message, HumanMessage)]
        if len(starts) <= self.max_turns:  # 此时本轮的用户消息已在状态里，所以保留 max_turns 条用户消息 = 历史 max_turns-1 轮 + 本轮
            return messages
        return messages[starts[-self.max_turns]:]

    def wrap_model_call(self, request, handler):
        return handler(request.override(messages=self.window(request.messages)))

    async def awrap_model_call(self, request, handler):
        return await handler(request.override(messages=self.window(request.messages)))


class ChineseModelCallLimit(ModelCallLimitMiddleware):
    """超出每轮模型调用上限时直接结束，并把官方的英文提示换成给用户看的中文提示。"""

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        update = super().before_model(state, runtime)
        if update and update.get("messages"):
            logger.warning("对话触达每轮模型调用上限（%s）", self.run_limit)
            update = {**update, "messages": [AIMessage(content=MODEL_LIMIT_MESSAGE)]}
        return update


class SessionRegistry:
    """会话登记：每会话一把锁（同会话串行）+ 会话级引用登记表。都只存在内存里，不做 TTL / 数量淘汰。

    条目只在有请求使用（或排队等锁）期间存在，最后一个使用者离开就丢弃；引用登记表在下一次使用时从库里重新加载，所以内存不会随会话数增长。
    """

    @dataclass
    class _Entry:
        lock: threading.Lock = field(default_factory=threading.Lock)
        users: int = 0  # 正在使用 / 排队等锁的请求数
        citations: CitationRegistry | None = None  # 拿到锁之后才加载（读库不放在全局 guard 里）

    def __init__(self, store: ChatStore):
        self.store = store
        self._guard = threading.Lock()
        self._entries: dict[str, SessionRegistry._Entry] = {}

    @contextmanager
    def session(self, session_id: str):
        """进入会话：持有会话锁到退出（同会话的并发请求在这里排队）；产出该会话的引用登记表。"""
        with self._guard:
            entry = self._entries.get(session_id)
            if entry is None:
                entry = self._entries[session_id] = self._Entry()
            entry.users += 1
        try:
            with entry.lock:
                if entry.citations is None:
                    entry.citations = CitationRegistry(self.store, session_id)
                yield entry.citations
        finally:
            with self._guard:
                entry.users -= 1
                if entry.users == 0 and self._entries.get(session_id) is entry:
                    del self._entries[session_id]

    def __len__(self) -> int:
        with self._guard:
            return len(self._entries)


@dataclass
class ChatResult:
    """一轮对话的结果：citations 是回答里实际出现 [n] 的片段 [(n, Document)]（按会话级编号解析，可能含历史轮次登记的）。"""

    session_id: str
    answer: str
    citations: list
    tool_calls: list[dict]


def _final_answer(messages) -> str:
    """最后一条不带 tool_calls 的 AI 消息的文本。"""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message.text.strip()
    return ""


def _tool_calls_of(messages) -> list[dict]:
    return [
        {"name": call["name"], "args": call["args"]}
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


def _current_turn(messages) -> list:
    """最后一条用户消息及之后的消息（= 本轮产生的消息）。"""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index:]
    return list(messages)


def incomplete_turn_start(messages) -> int | None:
    """状态末尾的那一轮不完整时返回这一轮（最后一条用户消息）的下标，完整则返回 None。

    不完整 = 最后一条用户消息之后，有 tool_calls 没对上 ToolMessage，或者最后一条不是不带 tool_calls 的 AI 回答。
    正常结束（含失败回滚）的状态不会这样；只有进程在一轮中途被杀才会留下这种状态。
    """
    start = next((index for index in range(len(messages) - 1, -1, -1) if isinstance(messages[index], HumanMessage)), None)
    if start is None:
        return None
    turn = messages[start:]
    last = turn[-1]
    if not isinstance(last, AIMessage) or last.tool_calls:
        return start
    answered = {message.tool_call_id for message in turn if isinstance(message, ToolMessage)}
    asked = {call["id"] for message in turn if isinstance(message, AIMessage) for call in message.tool_calls}
    return start if asked - answered else None


def render_history(messages, registry: CitationRegistry) -> list[dict]:
    """消息 -> [{role, content, tool_calls? / citations?}]。工具消息只给摘要；助手消息带 citations：[(n, Document)]，按会话级登记表解析其正文里标注的 [n]。"""
    result = []
    for message in messages:
        if isinstance(message, HumanMessage):
            result.append({"role": "user", "content": message.text})
        elif isinstance(message, AIMessage):
            entry = {"role": "assistant", "content": message.text, "citations": registry.cited_in(message.text)}
            if message.tool_calls:
                entry["tool_calls"] = [{"name": call["name"], "args": call["args"]} for call in message.tool_calls]
            result.append(entry)
        elif isinstance(message, ToolMessage):
            text = message.text.replace("\n", " ")
            result.append({"role": "tool", "content": f"[{message.name}] " + (text[:80] + "…" if len(text) > 80 else text)})
    return result


def prune_checkpoints(checkpointer, thread_id: str) -> None:
    """只保留这个 thread 最新的 checkpoint，删掉更早的及其 writes（避免无限保留会话时库随轮数膨胀）。

    默认的消息状态每个 checkpoint 都存完整的 channel_values，所以只留最新的一个不会丢任何消息。
    langgraph-checkpoint-sqlite 3.1.1 的 SqliteSaver 没有实现 prune / delete_for_runs（基类直接抛 NotImplementedError），所以直接对它的表执行 SQL；
    InMemorySaver 同样没有实现，内存模式（会话本来就随进程消失）跳过。
    """
    if not isinstance(checkpointer, SqliteSaver):
        return
    latest = (
        "(SELECT MAX(latest.checkpoint_id) FROM checkpoints AS latest "
        "WHERE latest.thread_id = {t}.thread_id AND latest.checkpoint_ns = {t}.checkpoint_ns)"
    )
    with checkpointer.cursor() as cursor:  # 持有 saver 的锁，退出时提交
        for table in ("writes", "checkpoints"):  # 先删 writes：它按 checkpoints 里的最新 id 比较，checkpoints 最后删
            cursor.execute(f"DELETE FROM {table} WHERE thread_id = ? AND checkpoint_id < {latest.format(t=table)}", (thread_id,))


class ChatService:
    """对 agent 的封装：会话管理、回滚与崩溃恢复、同步 / 流式两种调用。线程安全（每个会话内串行）。"""

    def __init__(self, agent, checkpointer, settings: Settings, store: ChatStore, clock=time.time):
        self.agent = agent
        self.checkpointer = checkpointer
        self.settings = settings
        self.store = store
        self.clock = clock
        self.sessions = SessionRegistry(store)
        # 一个循环约 5 个节点（含中间件钩子），默认的 25 步在模型调用上限较大时不够用
        self.recursion_limit = settings.agent_max_model_calls * 8 + 10

    def _config(self, session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}, "recursion_limit": self.recursion_limit}

    def _messages(self, session_id: str) -> list:
        return list(self.agent.get_state(self._config(session_id)).values.get("messages", []))

    def _restore(self, session_id: str, messages: list) -> None:
        """把会话的消息状态恢复成 messages；为空则整个 thread 删除（没有任何内容的会话不留 checkpoint）。"""
        if messages:
            self.agent.update_state(
                self._config(session_id), {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]}
            )
        else:
            self.checkpointer.delete_thread(session_id)

    def _prune(self, session_id: str) -> None:
        try:
            prune_checkpoints(self.checkpointer, session_id)
        except Exception:
            logger.exception("清理会话 %s 的旧 checkpoint 失败", session_id)

    def _begin_turn(self, session_id: str, registry: CitationRegistry) -> tuple[list, int]:
        """一轮开始（已持有会话锁）：先修复进程中途被杀留下的半截轮次，返回 (本轮之前的消息, 引用编号 mark)。"""
        messages = self._messages(session_id)
        start = incomplete_turn_start(messages)
        if start is not None:
            logger.warning("会话 %s 末尾有一轮没有正常结束（进程可能中途退出），已整轮丢弃", session_id)
            messages = messages[:start]
            self._restore(session_id, messages)
            registry.rollback(self.store.committed_citations(session_id))
            self._prune(session_id)
        return messages, registry.mark()

    def _rollback(self, session_id: str, before: list, registry: CitationRegistry, mark: int) -> None:
        """把会话恢复到本轮开始前的消息（引用登记表也恢复到 mark，库里的引用行同步撤销）；本轮之前是空会话则整体删除。
        会话摘要行只在首轮成功后才写入，所以首轮失败不会在 chat_sessions 里留下行。"""
        try:
            registry.rollback(mark)
        except Exception:
            logger.exception("回滚会话 %s 的引用编号失败", session_id)
        try:
            self._restore(session_id, before)
            self._prune(session_id)
        except Exception:
            logger.exception("回滚会话 %s 失败", session_id)

    def _commit_turn(self, session_id: str, message: str, messages: list, registry: CitationRegistry) -> None:
        """一轮成功结束：更新会话摘要（首轮插入、标题固定；message_count 累计用户消息 + 最终回答），然后清理旧 checkpoint。"""
        turn = _current_turn(messages)
        added = 1 + sum(1 for m in turn if isinstance(m, AIMessage) and not m.tool_calls)
        self.store.record_turn(session_id, message, added, registry.mark(), self.clock())
        self._prune(session_id)

    @staticmethod
    def _convert(exc: Exception) -> Exception:
        """agent 里抛出的异常 -> 与 /ask 一致的 GenerationError（超时 -> 504，其余 -> 502）；检索自身的错误原样抛出。"""
        if isinstance(exc, (GenerationError, ToolRetrievalError)):
            return exc
        if is_timeout_error(exc):
            return GenerationError(f"对话模型响应超时：{type(exc).__name__}", timeout=True)
        return GenerationError(f"对话模型调用失败：{type(exc).__name__}")

    @staticmethod
    def _inputs(message: str, sources: list[str] | None, registry: CitationRegistry):
        return {"messages": [HumanMessage(content=message)]}, ChatContext(registry=registry, default_sources=sources or None)

    def chat(self, session_id: str, message: str, sources: list[str] | None = None) -> ChatResult:
        """处理一条用户消息（非流式）。大模型失败抛 GenerationError，检索出错抛 ToolRetrievalError；失败的那一轮会被回滚。
        写库（会话摘要）失败也算本轮失败：回滚后原样抛出数据库异常（HTTP 500）。"""
        with self.sessions.session(session_id) as registry:
            payload, context = self._inputs(message, sources, registry)
            before, mark = self._begin_turn(session_id, registry)  # 在 try 之外：这一步自己失败时不能触发“回滚”（会把会话清空）
            finished, committing = False, False
            try:
                state = self.agent.invoke(payload, self._config(session_id), context=context)
                committing = True
                self._commit_turn(session_id, message, state["messages"], registry)
                finished = True
            except Exception as exc:
                if committing:
                    raise
                raise self._convert(exc) from exc
            finally:
                if not finished:
                    self._rollback(session_id, before, registry, mark)
        turn = _current_turn(state["messages"])
        answer = _final_answer(turn) or EMPTY_ANSWER_MESSAGE
        return ChatResult(session_id, answer, context.registry.cited_in(answer), _tool_calls_of(turn))

    def stream(self, session_id: str, message: str, sources: list[str] | None = None) -> Iterator[tuple[str, dict]]:
        """处理一条用户消息（流式），产出 (事件名, 数据)：
        session / tool_call{name,args} / token{text} / citations{citations:[(n, Document)]} / done{answer}。
        出错时抛异常（同 chat），由调用方转成 error 事件；消费方中途放弃（关闭生成器）也会回滚本轮。

        token 是模型每次生成里的文本增量，立即转发；若同一次生成随后又带了 tool_calls，则后续增量不再转发。
        部分模型会在调用工具前先输出一句话，这句话会混进 token 流，所以 done.answer 才是权威的最终回答。
        """
        yield "session", {"session_id": session_id}
        with self.sessions.session(session_id) as registry:
            payload, context = self._inputs(message, sources, registry)
            before, mark = self._begin_turn(session_id, registry)  # 在 try 之外：这一步自己失败时不能触发“回滚”（会把会话清空）
            finished, committing = False, False
            try:
                streamed, final, tool_steps = "", None, set()
                for mode, data in self.agent.stream(
                    payload, self._config(session_id), context=context, stream_mode=["messages", "updates"]
                ):
                    if mode == "messages":
                        chunk, metadata = data
                        if not isinstance(chunk, AIMessageChunk) or metadata.get("langgraph_node") != "model":
                            continue
                        if chunk.tool_call_chunks:
                            tool_steps.add(chunk.id)
                        if chunk.text and chunk.id not in tool_steps:
                            streamed += chunk.text
                            yield "token", {"text": chunk.text}
                    elif mode == "updates":
                        for update in data.values():
                            for new in (update or {}).get("messages", []) if isinstance(update, dict) else []:
                                if not isinstance(new, AIMessage):
                                    continue
                                if new.tool_calls:
                                    for call in new.tool_calls:
                                        yield "tool_call", {"name": call["name"], "args": call["args"]}
                                else:
                                    final = new
                answer = (final.text.strip() if final is not None else "") or EMPTY_ANSWER_MESSAGE
                if not streamed.strip():  # 没有逐字流出（例如触达模型调用上限时由中间件直接给出提示）：整段补发
                    yield "token", {"text": answer}
                committing = True
                self._commit_turn(session_id, message, self._messages(session_id), registry)
                finished = True
            except Exception as exc:
                if committing:
                    raise
                raise self._convert(exc) from exc
            finally:
                if not finished:
                    self._rollback(session_id, before, registry, mark)
        yield "citations", {"citations": context.registry.cited_in(answer)}
        yield "done", {"answer": answer}

    def history(self, session_id: str) -> list[dict] | None:
        """会话的全部消息（不受历史窗口影响）[{role, content, tool_calls? / citations?}]；会话不存在（含已删除）返回 None。"""
        if not self.store.exists(session_id):
            return None
        messages = self._messages(session_id)
        if not messages:
            return None
        return render_history(messages, CitationRegistry(self.store, session_id))

    def list_sessions(self) -> list[dict]:
        """会话摘要 [{session_id, title, updated_at(epoch 秒), message_count}]，按最近使用倒序；首轮没有成功的会话不列出。
        title 是第一条用户消息（压成单行，最多 TITLE_MAX_CHARS 字，之后固定）；message_count 累计用户消息和最终回答（不含工具调用步骤）。"""
        return self.store.list_sessions()

    def delete(self, session_id: str) -> bool:
        """删除会话的 checkpointer 数据、引用行和会话行（等当前请求结束后执行）；会话不存在返回 False。

        checkpointer 用独立连接，所以先删它、再在一个事务里删会话行和引用行；两步之间失败时会话行还在，再删一次即可删完。"""
        with self.sessions.session(session_id) as registry:
            existed = self.store.exists(session_id)
            self.checkpointer.delete_thread(session_id)
            self.store.delete_session(session_id)
            registry.clear()
        return existed

    def close(self) -> None:
        """关闭 SqliteSaver 独占的那条数据库连接（Database 本身由创建者关闭）。"""
        if isinstance(self.checkpointer, SqliteSaver):
            self.checkpointer.conn.close()


# ---------------------------------------------------------------- 对话 Agent 未就绪时的只读访问

def read_session_history(db: Database, store: ChatStore, session_id: str) -> list[dict] | None:
    """不依赖对话 Agent（大模型未配置 / 索引为空）：直接用一个 SqliteSaver 读 checkpoint 里的消息，引用编号从库里取。会话不存在返回 None。"""
    if not store.exists(session_id):
        return None
    saver = SqliteSaver(db.new_connection())
    try:
        checkpoint = saver.get_tuple({"configurable": {"thread_id": session_id}})
    finally:
        saver.conn.close()
    messages = list(checkpoint.checkpoint["channel_values"].get("messages", [])) if checkpoint else []
    if not messages:
        return None
    return render_history(messages, CitationRegistry(store, session_id))


def delete_session_offline(db: Database, store: ChatStore, session_id: str) -> bool:
    """不依赖对话 Agent 的删除：删 checkpointer 的 thread、引用行和会话行。此时没有对话请求在跑，所以不需要会话锁。会话不存在返回 False。"""
    if not store.exists(session_id):
        return False
    saver = SqliteSaver(db.new_connection())
    try:
        saver.delete_thread(session_id)
    finally:
        saver.conn.close()
    store.delete_session(session_id)
    return True


def _warn_deprecated_settings(settings: Settings) -> None:
    used = [name for name in DEPRECATED_SETTINGS if name in settings.model_fields_set]
    if used:
        logger.warning(
            "配置项 %s 已废弃、不再生效（%s）；请从 .env 中删除",
            "、".join(f"RAG_{name.upper()}" for name in used),
            "；".join(DEPRECATED_SETTINGS[name] for name in used),
        )


def build_chat_service(
    pipeline: RAGPipeline, settings: Settings, model: BaseChatModel | None = None, db: Database | None = None
) -> ChatService | None:
    """构建对话 Agent；大模型未配置返回 None。

    db 为 None：内存模式（InMemorySaver + 内存库，命令行 chat 与单元测试用）；否则会话持久化到 db 所在的数据库文件（SqliteSaver 用它的独立连接）。
    model 仅供测试注入（生产里模型只由 create_chat_model 创建）。
    """
    _warn_deprecated_settings(settings)
    model = model or create_chat_model(settings)
    if model is None:
        return None
    if db is None:
        checkpointer = InMemorySaver()
        store = ChatStore(Database(":memory:"))
    else:
        checkpointer = SqliteSaver(db.new_connection())
        checkpointer.setup()
        store = ChatStore(db)
    agent = create_agent(
        model=model,
        tools=build_tools(pipeline, settings),
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            HistoryTrimMiddleware(settings.chat_history_turns),
            ToolCallLimitMiddleware(run_limit=settings.agent_max_tool_calls, exit_behavior="continue"),
            ChineseModelCallLimit(run_limit=settings.agent_max_model_calls, exit_behavior="end"),
        ],
        context_schema=ChatContext,
        checkpointer=checkpointer,
    )
    return ChatService(agent, checkpointer, settings, store)
