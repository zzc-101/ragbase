"""对话 Agent：LangChain create_agent + 知识库工具 + 按会话的多轮记忆（进程内存）。

- 一次启动构建一次（build_chat_service）；会话 = LangGraph 的 thread，thread_id 就是 session_id，历史由 InMemorySaver 保存。
- 历史有界：每次调用开头把状态裁到最近 chat_history_turns 轮（一轮 = 一条用户消息及其后的全部回复 / 工具消息，
  整轮保留或丢弃，所以不会留下缺 ToolMessage 的 tool_calls）。不用 SummarizationMiddleware，避免额外的大模型调用。
- 会话数有上限、空闲有 TTL：超出时淘汰最久未用的空闲会话，并从 checkpointer 里删除它的 thread。
- 同一会话的请求串行（会话锁）；失败 / 客户端断开的那一轮会回滚，不会留下半截状态（否则下一轮的历史里可能有缺 ToolMessage 的 tool_calls，
  OpenAI 兼容服务会拒绝）。
- 限制：每轮工具调用上限（超出后工具被拒绝，模型据已有结果作答）、每轮模型调用上限（超出后直接结束并给出提示）。
"""
import logging
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware, hook_config
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, RemoveMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from agent_tools import ChatContext, ToolRetrievalError, build_tools
from config import Settings
from llm_factory import create_chat_model
from rag_pipeline import GenerationError, RAGPipeline, is_timeout_error

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是知识库问答助手，知识库收录了《系统架构设计师》《信息系统项目管理师》等教材。

规则：
1. 用户的问题涉及知识库里的专业内容（概念、方法、公式、流程、考点等）时，必须先调用 search_knowledge 检索，再回答。
2. 调用 search_knowledge 时，query 必须是自包含的完整问题：结合对话历史补全代词和省略，不要直接照抄“它”“那个”“上面说的”这类说法。
3. 只能依据工具返回的片段回答，不要补充片段里没有的事实、数据或常识；片段只覆盖问题的一部分时，只回答有依据的部分，并说明其余部分知识库里没有。
4. 在引用了某个片段的句子末尾用 [编号] 标注，编号取自工具返回的 [n]。只能使用本轮工具返回过的编号，不要自己编造；编号只在当次回答内有效，历史对话里出现过的编号不能沿用。
5. 工具表示没有找到可靠的相关内容时，回答“知识库中没有相关内容”，不要编造，也不要换着说法反复检索。
6. 寒暄、问候、关于你自己的问题可以直接回答，不需要调用工具。
7. 回答简洁，直接给结论和依据。"""

MODEL_LIMIT_MESSAGE = "本轮处理步骤过多，已停止。请把问题拆小一些，或换个问法再试。"
EMPTY_ANSWER_MESSAGE = "（模型没有给出回答，请重试）"
TITLE_MAX_CHARS = 30  # 会话列表里标题（第一条用户消息）的最大长度


class HistoryTrimMiddleware(AgentMiddleware):
    """每次调用开头把状态裁到最近 max_turns 轮；整轮保留或丢弃（从某条用户消息开始切），不会拆开 tool_calls 与 ToolMessage。"""

    def __init__(self, max_turns: int):
        super().__init__()
        self.max_turns = max_turns

    def before_agent(self, state, runtime):
        messages = state["messages"]
        starts = [index for index, message in enumerate(messages) if isinstance(message, HumanMessage)]
        if len(starts) <= self.max_turns:  # 此时本轮的用户消息已在状态里，所以保留 max_turns 条用户消息 = 历史 max_turns-1 轮 + 本轮
            return None
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages[starts[-self.max_turns]:]]}


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
    """会话登记：每会话一把锁（同会话串行）、最大会话数、空闲 TTL；被淘汰的会话调用 on_evict(session_id)。

    只淘汰“空闲”会话（没有请求持有或在排队等它的锁）；所有会话都忙时数量可暂时超过上限。
    """

    @dataclass
    class _Entry:
        lock: threading.Lock = field(default_factory=threading.Lock)
        users: int = 0  # 正在使用 / 排队等锁的请求数
        last_used: float = 0.0
        updated_at: float = 0.0  # 最近一次使用的挂钟时间（epoch 秒），给会话列表展示；TTL / LRU 仍用单调时钟

    def __init__(self, max_sessions: int, ttl: float, on_evict=None, clock=time.monotonic, wall_clock=time.time):
        self.max_sessions, self.ttl, self.on_evict, self.clock = max_sessions, ttl, on_evict, clock
        self.wall_clock = wall_clock
        self._guard = threading.Lock()
        self._entries: OrderedDict[str, SessionRegistry._Entry] = OrderedDict()  # 最久未用的在前

    def _expire_locked(self) -> None:
        now = self.clock()
        for session_id, entry in list(self._entries.items()):
            if entry.users == 0 and now - entry.last_used > self.ttl:
                self._drop_locked(session_id)

    def _make_room_locked(self) -> None:
        """新会话加入前，按 LRU 淘汰空闲会话，使加入后总数不超过上限。"""
        for session_id, entry in list(self._entries.items()):
            if len(self._entries) < self.max_sessions:
                break
            if entry.users == 0:
                self._drop_locked(session_id)

    def _drop_locked(self, session_id: str) -> None:
        del self._entries[session_id]
        logger.info("淘汰会话 %s", session_id)
        if self.on_evict:
            try:
                self.on_evict(session_id)
            except Exception:
                logger.exception("清理会话 %s 的历史失败", session_id)

    @contextmanager
    def session(self, session_id: str):
        """进入会话：登记 / 续期，并持有会话锁到退出（同会话的并发请求在这里排队）。"""
        with self._guard:
            self._expire_locked()
            entry = self._entries.get(session_id)
            if entry is None:
                self._make_room_locked()
                entry = self._entries[session_id] = self._Entry()
            entry.users += 1
            entry.last_used = self.clock()
            entry.updated_at = self.wall_clock()
            self._entries.move_to_end(session_id)
        try:
            with entry.lock:
                yield
        finally:
            with self._guard:
                entry.users -= 1
                entry.last_used = self.clock()
                entry.updated_at = self.wall_clock()
                self._entries.move_to_end(session_id)

    def known(self, session_id: str) -> bool:
        """会话是否还在登记表里（先清理过期的，不续期）。"""
        with self._guard:
            self._expire_locked()
            return session_id in self._entries

    def snapshot(self) -> list[tuple[str, float]]:
        """登记表里所有会话的 [(session_id, 最近使用的挂钟时间)]（先清理过期的，不续期）。"""
        with self._guard:
            self._expire_locked()
            return [(session_id, entry.updated_at) for session_id, entry in self._entries.items()]

    def __len__(self) -> int:
        with self._guard:
            return len(self._entries)


@dataclass
class ChatResult:
    """一轮对话的结果：citations 是本轮回答里实际出现 [n] 的片段 [(n, Document)]。"""

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


class ChatService:
    """对 agent 的封装：会话管理、回滚、同步 / 流式两种调用。线程安全（每个会话内串行）。"""

    def __init__(self, agent, checkpointer: InMemorySaver, settings: Settings, clock=time.monotonic):
        self.agent = agent
        self.checkpointer = checkpointer
        self.settings = settings
        self.sessions = SessionRegistry(
            settings.chat_max_sessions, settings.chat_session_ttl, on_evict=checkpointer.delete_thread, clock=clock
        )
        # 一个循环约 5 个节点（含中间件钩子），默认的 25 步在模型调用上限较大时不够用
        self.recursion_limit = settings.agent_max_model_calls * 8 + 10

    def _config(self, session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}, "recursion_limit": self.recursion_limit}

    def _messages(self, session_id: str) -> list:
        return list(self.agent.get_state(self._config(session_id)).values.get("messages", []))

    def _rollback(self, session_id: str, before: list) -> None:
        """把会话恢复到本轮开始前的消息；本轮之前是空会话则整体删除。"""
        try:
            if before:
                self.agent.update_state(
                    self._config(session_id), {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *before]}
                )
            else:
                self.checkpointer.delete_thread(session_id)
        except Exception:
            logger.exception("回滚会话 %s 失败", session_id)

    @staticmethod
    def _convert(exc: Exception) -> Exception:
        """agent 里抛出的异常 -> 与 /ask 一致的 GenerationError（超时 -> 504，其余 -> 502）；检索自身的错误原样抛出。"""
        if isinstance(exc, (GenerationError, ToolRetrievalError)):
            return exc
        if is_timeout_error(exc):
            return GenerationError(f"对话模型响应超时：{type(exc).__name__}", timeout=True)
        return GenerationError(f"对话模型调用失败：{type(exc).__name__}")

    def _inputs(self, message: str, sources: list[str] | None):
        return {"messages": [HumanMessage(content=message)]}, ChatContext(default_sources=sources or None)

    def chat(self, session_id: str, message: str, sources: list[str] | None = None) -> ChatResult:
        """处理一条用户消息（非流式）。大模型失败抛 GenerationError，检索出错抛 ToolRetrievalError；失败的那一轮会被回滚。"""
        payload, context = self._inputs(message, sources)
        with self.sessions.session(session_id):
            before, finished = self._messages(session_id), False
            try:
                state = self.agent.invoke(payload, self._config(session_id), context=context)
                finished = True
            except Exception as exc:
                raise self._convert(exc) from exc
            finally:
                if not finished:
                    self._rollback(session_id, before)
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
        payload, context = self._inputs(message, sources)
        yield "session", {"session_id": session_id}
        with self.sessions.session(session_id):
            before, finished = self._messages(session_id), False
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
                finished = True
            except Exception as exc:
                raise self._convert(exc) from exc
            finally:
                if not finished:
                    self._rollback(session_id, before)
        yield "citations", {"citations": context.registry.cited_in(answer)}
        yield "done", {"answer": answer}

    def history(self, session_id: str) -> list[dict] | None:
        """会话消息 [{role, content, tool_calls?}]；会话不存在（含已被清空 / 淘汰）返回 None。工具消息只给摘要。"""
        if not self.sessions.known(session_id):
            return None
        messages = self._messages(session_id)
        if not messages:
            return None
        result = []
        for message in messages:
            if isinstance(message, HumanMessage):
                result.append({"role": "user", "content": message.text})
            elif isinstance(message, AIMessage):
                entry = {"role": "assistant", "content": message.text}
                if message.tool_calls:
                    entry["tool_calls"] = [{"name": call["name"], "args": call["args"]} for call in message.tool_calls]
                result.append(entry)
            elif isinstance(message, ToolMessage):
                text = message.text.replace("\n", " ")
                result.append({"role": "tool", "content": f"[{message.name}] " + (text[:80] + "…" if len(text) > 80 else text)})
        return result

    def list_sessions(self) -> list[dict]:
        """会话摘要 [{session_id, title, updated_at(epoch 秒), message_count}]，按最近使用倒序；空会话（已清空 / 回滚）不列出。
        title 取第一条用户消息（压成单行，最多 TITLE_MAX_CHARS 字）；message_count 只数用户消息和最终回答（不含工具调用步骤）。"""
        result = []
        for session_id, updated_at in self.sessions.snapshot():
            try:
                messages = self._messages(session_id)
            except Exception:
                logger.exception("读取会话 %s 失败", session_id)
                continue
            users = [m for m in messages if isinstance(m, HumanMessage)]
            if not users:
                continue
            title = " ".join(users[0].text.split())
            if len(title) > TITLE_MAX_CHARS:
                title = title[:TITLE_MAX_CHARS] + "…"
            count = len(users) + sum(1 for m in messages if isinstance(m, AIMessage) and not m.tool_calls)
            result.append({"session_id": session_id, "title": title, "updated_at": updated_at, "message_count": count})
        result.sort(key=lambda item: item["updated_at"], reverse=True)
        return result

    def delete(self, session_id: str) -> bool:
        """清空会话历史（等当前请求结束）；会话不存在返回 False。登记项留给 TTL / LRU 淘汰。"""
        if not self.sessions.known(session_id):
            return False
        with self.sessions.session(session_id):
            existed = bool(self._messages(session_id))
            self.checkpointer.delete_thread(session_id)
        return existed


def build_chat_service(pipeline: RAGPipeline, settings: Settings, model: BaseChatModel | None = None) -> ChatService | None:
    """构建对话 Agent；大模型未配置返回 None。model 仅供测试注入（生产里模型只由 create_chat_model 创建）。"""
    model = model or create_chat_model(settings)
    if model is None:
        return None
    checkpointer = InMemorySaver()
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
    return ChatService(agent, checkpointer, settings)
