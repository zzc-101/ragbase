"""表结构版本与迁移。

- 版本号存在 SQLite 的 `PRAGMA user_version` 里：0 表示全新（或没有任何迁移）的库。
- MIGRATIONS 是按顺序排列的迁移函数，第 i 个（从 0 数）把版本从 i 升到 i+1；只增不改，已发布的迁移不要再动，要改表结构就追加新的迁移。
- 每个迁移各自在一个事务（BEGIN IMMEDIATE）里执行，迁移与版本号的更新一起提交：中途失败整个迁移回滚，版本号不变，下次启动从这一步继续。
- 迁移函数只能用 `conn.execute(...)`，不要用 `executescript`（它会隐式提交，破坏事务）。
- LangGraph checkpointer 的表由它自己的 `setup()` 管理，不在这里。
"""
import sqlite3
from collections.abc import Callable


def _migration_1_meta(conn: sqlite3.Connection) -> None:
    """键值元数据表：存“某次一次性导入是否做过”之类的标记。"""
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")


def _migration_2_chat(conn: sqlite3.Connection) -> None:
    """对话会话：会话摘要表 + 会话级引用编号表（checkpointer 的表由 LangGraph 自己建，不在这里）。

    committed_citations 是“已提交编号”：该会话最后一个完整轮次之后引用表里的最大编号，崩溃恢复时据此清掉半截轮次多登记的引用行。
    chat_citations.document 是 JSON：{"page_content": …, "metadata": {…}}。
    """
    conn.execute(
        "CREATE TABLE chat_sessions ("
        "session_id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL, "
        "message_count INTEGER NOT NULL, committed_citations INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute("CREATE INDEX idx_chat_sessions_updated ON chat_sessions (updated_at DESC)")
    conn.execute(
        "CREATE TABLE chat_citations ("
        "session_id TEXT NOT NULL, number INTEGER NOT NULL, chunk_key TEXT NOT NULL, document TEXT NOT NULL, "
        "PRIMARY KEY (session_id, number), UNIQUE (session_id, chunk_key))"
    )


def _migration_3_diagnose(conn: sqlite3.Connection) -> None:
    """评测题集 + 体检报告历史。

    eval_questions.reference 可空：空时这题只计耗时、不计召回率（ragbase eval 会跳过它）。
    diagnose_reports 的 params / summary / recommendations / records / extra 都是 JSON 文本，内容取自 build_report 的对应部分；
    extra 存 question_ids、question_digest、concurrent_chat_requests、warmup_question。
    """
    conn.execute(
        "CREATE TABLE eval_questions ("
        "id INTEGER PRIMARY KEY, question TEXT NOT NULL, reference TEXT, difficulty TEXT, type TEXT, source_topic TEXT, "
        "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE diagnose_reports ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, origin TEXT NOT NULL CHECK(origin IN ('cli','web')), "
        "mode TEXT NOT NULL, question_count INTEGER NOT NULL, params TEXT NOT NULL, summary TEXT NOT NULL, "
        "recommendations TEXT NOT NULL, records TEXT NOT NULL, extra TEXT NOT NULL)"
    )


MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [
    _migration_1_meta,
    _migration_2_chat,
    _migration_3_diagnose,
]


def current_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def migrate(conn: sqlite3.Connection) -> int:
    """把数据库迁移到最新版本，返回迁移后的版本号。

    已经是最新（或更新，比如被新版本程序升级过）时什么也不做。多个进程同时启动时，靠 BEGIN IMMEDIATE 串行，
    拿到写锁后重新读取版本号，已经被别的进程升过的步骤会跳过。
    """
    previous = conn.isolation_level
    conn.isolation_level = None  # 事务完全由本函数手动管理
    try:
        for index, step in enumerate(MIGRATIONS):
            if current_version(conn) > index:
                continue  # 快速路径：这一步早就做过
            conn.execute("BEGIN IMMEDIATE")
            try:
                if current_version(conn) > index:  # 拿到锁后再确认一次（别的进程可能刚升过）
                    conn.execute("ROLLBACK")
                    continue
                step(conn)
                conn.execute(f"PRAGMA user_version = {index + 1}")  # PRAGMA 不支持参数绑定；index 是我们自己的整数
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        return current_version(conn)
    finally:
        conn.isolation_level = previous
