"""内置 SQLite 数据库：连接、事务、元数据与打开入口。

- 一个 Database 对象持有一条主连接（check_same_thread=False，供线程池里的多个请求共用），
  用一把进程内可重入锁串行化“主连接上的所有读写”；写事务用 BEGIN IMMEDIATE，一开始就拿写锁，避免读转写时的 SQLITE_BUSY。
- 文件库开启 WAL（读写互不阻塞）、foreign_keys=ON、busy_timeout=5000ms（别的连接——比如 LangGraph checkpointer 的独立连接——
  正在写时最多等 5 秒，而不是立刻抛 database is locked）。
- ":memory:" 表示内存库（命令行 chat 和测试用）：不开 WAL，不支持 new_connection()。
- 构造时自动迁移到最新表结构（见 migrations.py）；任何打不开 / 文件损坏 / 版本过新的情况都抛 DatabaseError（中文说明）。
"""
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ragbase.config import Settings, db_file, eval_questions_file
from ragbase.db.migrations import MIGRATIONS, migrate

MEMORY = ":memory:"
BUSY_TIMEOUT_MS = 5000


class DatabaseError(RuntimeError):
    """数据库打不开 / 文件损坏 / 版本过新等问题；消息是给用户看的中文说明。"""


def _connect(target: str, memory: bool, **kwargs) -> sqlite3.Connection:
    conn = sqlite3.connect(target, check_same_thread=False, timeout=BUSY_TIMEOUT_MS / 1000, **kwargs)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys=ON")
        if not memory:
            conn.execute("PRAGMA journal_mode=WAL")  # 首次访问文件时才会真正读它：损坏的文件会在这里报错
    except BaseException:
        conn.close()
        raise
    return conn


class Database:
    def __init__(self, path: Path | str):
        self._memory = str(path) == MEMORY
        self.path: Path | str = MEMORY if self._memory else Path(path)
        self._lock = threading.RLock()
        self._closed = False
        try:
            # isolation_level=None：自动提交模式，事务由 transaction() 手动 BEGIN / COMMIT
            self._conn = _connect(MEMORY if self._memory else str(self.path), self._memory, isolation_level=None)
        except sqlite3.Error as exc:
            raise DatabaseError(self._describe("无法打开数据库", exc)) from exc
        try:
            version = migrate(self._conn)
        except sqlite3.Error as exc:
            self._conn.close()
            raise DatabaseError(self._describe("数据库迁移失败", exc)) from exc
        except BaseException:
            self._conn.close()
            raise
        if version > len(MIGRATIONS):
            self._conn.close()
            raise DatabaseError(
                f"数据库 {self.path} 的版本（{version}）比当前程序支持的（{len(MIGRATIONS)}）更新：请升级 ragbase，不要用旧版本打开新版本的数据库"
            )

    def _describe(self, action: str, exc: Exception) -> str:
        hint = "请检查目录权限、磁盘空间；若文件已损坏，可备份后删除它（会丢失会话与体检记录），重启后自动重建"
        return f"{action}（{self.path}）：{exc}。{hint}"

    # ---------------------------------------------------------------- 读写

    def query(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        """执行一条语句并返回全部行（主连接，串行）。写入请用 transaction()。"""
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """写事务：进程内写锁 + BEGIN IMMEDIATE；正常退出提交，异常回滚并原样抛出。

        同一线程可以在事务里调用 query()（锁可重入），读到的是事务内的最新状态。不要嵌套 transaction()。
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

    # ---------------------------------------------------------------- 元数据与版本

    def meta_get(self, key: str) -> str | None:
        rows = self.query("SELECT value FROM meta WHERE key = ?", (key,))
        return rows[0]["value"] if rows else None

    def meta_set(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        """在调用方的事务里写一条元数据：with db.transaction() as conn: db.meta_set(conn, k, v)。"""
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def schema_version(self) -> int:
        return self.query("PRAGMA user_version")[0][0]

    # ---------------------------------------------------------------- 其它连接与关闭

    def new_connection(self) -> sqlite3.Connection:
        """同一个数据库文件的另一条独立连接（给 LangGraph checkpointer 用），保持 sqlite3 默认的事务模式（它自己 commit）。"""
        if self._memory:
            raise DatabaseError("内存数据库不支持再开新连接（命令行 chat 用 InMemorySaver，不需要它）")
        try:
            return _connect(str(self.path), False)
        except sqlite3.Error as exc:
            raise DatabaseError(self._describe("无法打开数据库连接", exc)) from exc

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._conn.close()


def open_database(settings: Settings | None = None) -> Database:
    """按配置打开数据库：创建父目录、打开、迁移到最新。目录不可写或文件损坏时抛 DatabaseError（中文说明）。

    打开后会做一次“首次导入”：库里没有导入标记、题集表为空且 RAG_HOME/eval_questions.json 存在时，把它导入题集表（只做一次）。
    """
    path = db_file(settings)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DatabaseError(f"无法创建数据库目录 {path.parent}：{exc}。请检查目录权限，或用 RAG_DB_PATH 指向可写位置") from exc
    db = Database(path)
    try:
        # 函数内导入：evaluation 包不能在 db 模块加载时被拉进来（避免循环导入，也保持 db 层轻量）
        from ragbase.evaluation.questions import ensure_initial_import

        ensure_initial_import(db, eval_questions_file())
    except sqlite3.Error as exc:
        db.close()
        raise DatabaseError(db._describe("题集首次导入失败", exc)) from exc
    except BaseException:
        db.close()
        raise
    return db
