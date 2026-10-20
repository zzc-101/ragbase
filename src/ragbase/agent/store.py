"""对话会话的持久化：会话摘要（chat_sessions）与会话级引用编号（chat_citations），存在内置 SQLite 里。

- 消息历史本身在 LangGraph 的 checkpointer 里（同一个数据库文件，表由 SqliteSaver.setup() 建）；这里只存列表页要用的摘要和引用。
- chat_sessions 的行在会话“第一轮成功结束”时才插入，之后每轮成功结束更新；所以首轮失败 / 回滚的会话不会出现在列表里。
- committed_citations（已提交编号）是最后一个完整轮次之后引用表里的最大编号：进程在一轮中途被杀时，引用表可能多出这一轮登记的行，
  下一轮开始时按它截断即可（见 ChatService 的半截轮次恢复）。
- 所有写操作走 Database.transaction()（进程内写锁 + BEGIN IMMEDIATE），读走 Database.query()。
"""
import json
import logging
from typing import Any

from langchain_core.documents import Document

from ragbase.db import Database

logger = logging.getLogger(__name__)

TITLE_MAX_CHARS = 30  # 会话列表里标题（第一条用户消息）的最大长度


def make_title(message: str) -> str:
    """第一条用户消息压成单行，最多 TITLE_MAX_CHARS 字。"""
    title = " ".join(message.split())
    return title[:TITLE_MAX_CHARS] + "…" if len(title) > TITLE_MAX_CHARS else title


def _json_default(value: Any) -> Any:
    """元数据里偶尔会有 numpy 数值之类的非 JSON 类型：能转数字就转数字，否则转字符串，保证写库不会因此失败。"""
    for convert in (int, float):
        try:
            if convert(value) == value:
                return convert(value)
        except (TypeError, ValueError):
            continue
    return str(value)


def dump_document(doc: Document) -> str:
    return json.dumps({"page_content": doc.page_content, "metadata": doc.metadata}, ensure_ascii=False, default=_json_default)


def load_document(text: str) -> Document:
    data = json.loads(text)
    return Document(page_content=data["page_content"], metadata=data.get("metadata") or {})


class ChatStore:
    """会话摘要与引用编号的读写。线程安全（靠 Database 的锁）。"""

    def __init__(self, db: Database):
        self.db = db

    # ---------------------------------------------------------------- 会话摘要

    def record_turn(self, session_id: str, first_message: str, added_messages: int, committed_citations: int, now: float) -> None:
        """一轮成功结束后调用：首轮插入（标题取首条用户消息并固定），之后累加 message_count、更新 updated_at 和已提交编号。"""
        with self.db.transaction() as conn:
            updated = conn.execute(
                "UPDATE chat_sessions SET message_count = message_count + ?, updated_at = ?, committed_citations = ? WHERE session_id = ?",
                (added_messages, now, committed_citations, session_id),
            ).rowcount
            if not updated:
                conn.execute(
                    "INSERT INTO chat_sessions (session_id, title, created_at, updated_at, message_count, committed_citations) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, make_title(first_message), now, now, added_messages, committed_citations),
                )

    def list_sessions(self) -> list[dict]:
        """[{session_id, title, updated_at(epoch 秒), message_count}]，按最近使用倒序。"""
        rows = self.db.query(
            "SELECT session_id, title, updated_at, message_count FROM chat_sessions ORDER BY updated_at DESC, session_id"
        )
        return [dict(row) for row in rows]

    def exists(self, session_id: str) -> bool:
        return bool(self.db.query("SELECT 1 FROM chat_sessions WHERE session_id = ?", (session_id,)))

    def committed_citations(self, session_id: str) -> int:
        """已提交编号；会话不存在（首轮还没成功结束）为 0。"""
        rows = self.db.query("SELECT committed_citations FROM chat_sessions WHERE session_id = ?", (session_id,))
        return rows[0]["committed_citations"] if rows else 0

    def delete_session(self, session_id: str) -> bool:
        """在一个事务里删掉会话行和它的全部引用行；会话行存在返回 True。"""
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM chat_citations WHERE session_id = ?", (session_id,))
            return conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,)).rowcount > 0

    # ---------------------------------------------------------------- 引用编号

    def load_citations(self, session_id: str) -> list[tuple[int, str, Document]]:
        """[(编号, chunk_key, 片段)]，按编号升序。"""
        rows = self.db.query(
            "SELECT number, chunk_key, document FROM chat_citations WHERE session_id = ? ORDER BY number", (session_id,)
        )
        return [(row["number"], row["chunk_key"], load_document(row["document"])) for row in rows]

    def add_citation(self, session_id: str, number: int, chunk_key: str, doc: Document) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO chat_citations (session_id, number, chunk_key, document) VALUES (?, ?, ?, ?)",
                (session_id, number, chunk_key, dump_document(doc)),
            )

    def truncate_citations(self, session_id: str, keep: int) -> None:
        """删掉编号大于 keep 的引用行（编号按登记顺序连续分配，所以保留的永远是前 keep 个）。"""
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM chat_citations WHERE session_id = ? AND number > ?", (session_id, keep))
