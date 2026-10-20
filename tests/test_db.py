"""内置 SQLite 数据库基础（ragbase.db）的单元测试：建库 / 迁移 / 事务 / 并发写 / 错误信息 / 配置 / doctor 检查。

全部使用临时目录里的数据库文件（或内存库）：不碰项目里真实的 ragbase.db，不加载模型，不调用大模型。

运行：uv run python tests/test_db.py
"""
import contextlib
import os
import sqlite3
import tempfile
import threading
from pathlib import Path

from ragbase import doctor
from ragbase.config import Settings, db_file, get_settings, project_home
from ragbase.db import MIGRATIONS, Database, DatabaseError, migrate, open_database
from ragbase.db import migrations as migrations_module

_SCRATCH = tempfile.TemporaryDirectory(prefix="ragdemo-db-test-")
SCRATCH = Path(_SCRATCH.name).resolve()  # resolve：macOS 的 /var 是 /private/var 的软链
_counter = iter(range(10_000))


def _fresh_dir(name: str) -> Path:
    path = SCRATCH / f"{name}-{next(_counter)}"
    path.mkdir()
    return path


@contextlib.contextmanager
def env(**values):
    """临时设置 / 删除环境变量（None = 删除），并先清掉所有 RAG_* / ARK_* 变量，保证与外部环境无关。"""
    saved = dict(os.environ)
    for key in list(os.environ):
        if key.startswith(("RAG_", "ARK_")):
            del os.environ[key]
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    get_settings.cache_clear()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
        get_settings.cache_clear()


def _open(home: Path, **env_values) -> Database:
    """在“RAG_HOME=home”的环境里按配置打开数据库（db_file() 在调用时读环境变量，必须在 env 里调用，否则会落到当前目录）。"""
    with env(RAG_HOME=str(home), **env_values):
        return open_database(Settings(_env_file=None))


def _raw_version(path: Path) -> int:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def _tables(path: Path) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        conn.close()


def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


# ---------------------------------------------------------------- 建库与迁移

def test_fresh_file_is_created_and_migrated():
    home = _fresh_dir("fresh")
    db = _open(home)
    try:
        assert (home / "ragbase.db").is_file()
        assert db.path == home / "ragbase.db"
        assert db.schema_version() == len(MIGRATIONS) >= 1
        assert db.meta_get("missing") is None
        with db.transaction() as conn:
            db.meta_set(conn, "k", "v1")
        assert db.meta_get("k") == "v1"
        with db.transaction() as conn:
            db.meta_set(conn, "k", "v2")  # 覆盖写
        assert db.meta_get("k") == "v2"
        # 连接参数
        assert db.query("PRAGMA journal_mode")[0][0] == "wal"
        assert db.query("PRAGMA foreign_keys")[0][0] == 1
        assert db.query("PRAGMA busy_timeout")[0][0] == 5000
        assert db.query("SELECT key, value FROM meta")[0]["key"] == "k"  # row_factory=sqlite3.Row
    finally:
        db.close()


def test_open_database_creates_missing_parent_dirs():
    home = _fresh_dir("parents")
    target = home / "a" / "b" / "my.db"
    db = _open(home, RAG_DB_PATH=str(target))
    try:
        assert target.is_file()
    finally:
        db.close()


def test_reopen_is_idempotent():
    path = _fresh_dir("reopen") / "x.db"
    db = Database(path)
    with db.transaction() as conn:
        db.meta_set(conn, "keep", "me")
    version = db.schema_version()
    db.close()
    db = Database(path)
    try:
        assert db.schema_version() == version == len(MIGRATIONS)
        assert db.meta_get("keep") == "me"
        assert migrate(db._conn) == version  # 已是最新：再迁移什么也不做
    finally:
        db.close()


def test_upgrade_from_older_version():
    path = _fresh_dir("upgrade") / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 0")
    conn.execute("CREATE TABLE unrelated (id INTEGER)")  # 非空但没有版本号的旧库：迁移不动别的表
    conn.commit()
    conn.close()
    assert _raw_version(path) == 0
    db = Database(path)
    try:
        assert db.schema_version() == len(MIGRATIONS)
        assert db.meta_get("anything") is None  # meta 表已建好
        assert "unrelated" in _tables(path)
    finally:
        db.close()


def test_failed_migration_rolls_back_and_keeps_version():
    path = _fresh_dir("badmig") / "m.db"
    Database(path).close()
    before = _raw_version(path)

    def broken(conn):
        conn.execute("CREATE TABLE half_done (id INTEGER)")
        raise RuntimeError("迁移故意失败")

    MIGRATIONS.append(broken)
    try:
        try:
            Database(path)
            raise AssertionError("迁移失败应该抛出异常")
        except RuntimeError as exc:
            assert "故意失败" in str(exc)
    finally:
        MIGRATIONS.remove(broken)
    assert _raw_version(path) == before
    assert "half_done" not in _tables(path)
    Database(path).close()  # 去掉坏迁移后仍能正常打开


def test_newer_database_version_is_rejected():
    path = _fresh_dir("newer") / "n.db"
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {len(MIGRATIONS) + 5}")
    conn.commit()
    conn.close()
    try:
        Database(path)
        raise AssertionError("版本过新应该抛 DatabaseError")
    except DatabaseError as exc:
        assert "数据库" in str(exc) and "升级" in str(exc)


def test_corrupt_file_gives_chinese_error():
    home = _fresh_dir("corrupt")
    (home / "ragbase.db").write_bytes(b"this is definitely not a sqlite database file" * 50)
    try:
        _open(home)
        raise AssertionError("损坏的文件应该抛 DatabaseError")
    except DatabaseError as exc:
        assert "数据库" in str(exc) and str(home / "ragbase.db") in str(exc)


# ---------------------------------------------------------------- 事务与并发

def test_transaction_rolls_back_on_error():
    db = Database(_fresh_dir("rollback") / "r.db")
    try:
        try:
            with db.transaction() as conn:
                db.meta_set(conn, "ghost", "boo")
                assert db.meta_get("ghost") == "boo"  # 事务内同线程能读到自己的写入
                raise ValueError("中途出错")
        except ValueError:
            pass
        assert db.meta_get("ghost") is None
        with db.transaction() as conn:  # 回滚后连接仍可正常使用
            db.meta_set(conn, "real", "ok")
        assert db.meta_get("real") == "ok"
    finally:
        db.close()


def test_concurrent_writers_do_not_raise_locked():
    db = Database(_fresh_dir("concurrent") / "c.db")
    with db.transaction() as conn:
        conn.execute("CREATE TABLE rows (src TEXT NOT NULL, n INTEGER NOT NULL)")
    errors: list[BaseException] = []
    per_thread, workers = 50, 8
    start = threading.Barrier(workers + 1)

    def writer(index: int):
        try:
            start.wait()
            for n in range(per_thread):
                with db.transaction() as conn:
                    conn.execute("INSERT INTO rows VALUES (?, ?)", (f"w{index}", n))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def other_connection_writer():
        """模拟 LangGraph checkpointer：另一条连接，自己的事务，与主连接的写入互相争用。"""
        conn = db.new_connection()
        try:
            start.wait()
            for n in range(per_thread):
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("INSERT INTO rows VALUES (?, ?)", ("saver", n))
                conn.commit()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(workers)]
    threads.append(threading.Thread(target=other_connection_writer))
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        assert not any(thread.is_alive() for thread in threads), "写线程卡死"
        assert not errors, f"并发写出现异常：{errors[:3]}"
        assert db.query("SELECT count(*) FROM rows")[0][0] == (workers + 1) * per_thread
        assert db.query("SELECT count(*) FROM rows WHERE src = 'saver'")[0][0] == per_thread
    finally:
        db.close()


def test_new_connection_shares_the_file():
    db = Database(_fresh_dir("newconn") / "s.db")
    conn = db.new_connection()
    try:
        with db.transaction() as main:
            db.meta_set(main, "shared", "yes")
        assert conn.execute("SELECT value FROM meta WHERE key = 'shared'").fetchone()["value"] == "yes"
    finally:
        conn.close()
        db.close()


# ---------------------------------------------------------------- 内存库

def test_memory_database():
    db = Database(":memory:")
    try:
        assert db.path == ":memory:"
        assert db.schema_version() == len(MIGRATIONS)
        with db.transaction() as conn:
            db.meta_set(conn, "a", "b")
        assert db.meta_get("a") == "b"
        assert db.query("PRAGMA journal_mode")[0][0] == "memory"  # 内存库不开 WAL
        try:
            db.new_connection()
            raise AssertionError("内存库不应支持 new_connection")
        except DatabaseError as exc:
            assert "内存" in str(exc)
    finally:
        db.close()
    db.close()  # 重复关闭不报错
    assert not Path(":memory:").exists(), "内存库不应在当前目录留下文件"


# ---------------------------------------------------------------- 错误信息

def test_unwritable_dir_gives_chinese_error():
    home = _fresh_dir("readonly")
    locked = home / "locked"
    locked.mkdir()
    os.chmod(locked, 0o555)
    try:
        probe = locked / ".probe"
        try:
            probe.touch()
        except OSError:
            pass
        else:
            probe.unlink()
            print("跳过只读目录的断言：当前用户（root？）无视目录权限，目录仍然可写")
            return
        try:
            _open(home, RAG_DB_PATH=str(locked / "ragbase.db"))
            raise AssertionError("不可写目录应该抛 DatabaseError")
        except DatabaseError as exc:
            assert "数据库" in str(exc), str(exc)
        # 连父目录都建不出来的情况
        try:
            _open(home, RAG_DB_PATH=str(locked / "sub" / "ragbase.db"))
            raise AssertionError("无法创建目录应该抛 DatabaseError")
        except DatabaseError as exc:
            assert "数据库" in str(exc), str(exc)
    finally:
        os.chmod(locked, 0o755)


# ---------------------------------------------------------------- 配置

def test_db_path_setting():
    home = _fresh_dir("setting")
    with env(RAG_HOME=str(home)):
        assert Settings(_env_file=None).db_path is None
        assert db_file() == project_home() / "ragbase.db" == home / "ragbase.db"
    with env(RAG_HOME=str(home), RAG_DB_PATH=""):  # 空值视为未设置
        assert Settings(_env_file=None).db_path is None
        assert db_file() == home / "ragbase.db"
        assert db_file(Settings(_env_file=None)) == home / "ragbase.db"
    with env(RAG_HOME=str(home), RAG_DB_PATH="   "):
        assert Settings(_env_file=None).db_path is None
    custom = home / "elsewhere" / "my.db"
    with env(RAG_HOME=str(home), RAG_DB_PATH=str(custom)):
        assert Settings(_env_file=None).db_path == custom
        assert db_file() == custom
        assert db_file(Settings(_env_file=None)) == custom


# ---------------------------------------------------------------- doctor 检查（只读，不建库）

def _db_check(home: Path, **env_values) -> doctor.Check:
    with env(RAG_HOME=str(home), **env_values):
        checks = doctor.check_database(Settings(_env_file=None))
    assert len(checks) == 1, checks
    return checks[0]


def test_doctor_database_check():
    home = _fresh_dir("doctor-db")

    # 还没有数据库：只提示，不创建文件
    check = _db_check(home)
    assert check.level == doctor.OK and "尚未创建" in check.text, check
    assert not (home / "ragbase.db").exists(), "doctor 不应创建数据库"

    # 目录都不存在（且连上级也建不出来之前）：仍不创建任何目录
    assert _db_check(home, RAG_DB_PATH=str(home / "no" / "such" / "x.db")).level == doctor.OK
    assert not (home / "no").exists()

    # 健康的最新库
    Database(home / "ragbase.db").close()
    check = _db_check(home)
    assert check.level == doctor.OK and "最新" in check.text, check

    # 旧版本：警告，提示 serve 会自动升级
    conn = sqlite3.connect(home / "ragbase.db")
    conn.execute("PRAGMA user_version = 0")
    conn.close()
    check = _db_check(home)
    assert check.level == doctor.WARN and "自动升级" in check.hint, check
    assert _raw_version(home / "ragbase.db") == 0, "doctor 不应迁移数据库"

    # 版本过新：不通过
    conn = sqlite3.connect(home / "ragbase.db")
    conn.execute(f"PRAGMA user_version = {len(MIGRATIONS) + 3}")
    conn.close()
    check = _db_check(home)
    assert check.level == doctor.FAIL and "更新" in check.text, check

    # 文件损坏：不通过
    bad_home = _fresh_dir("doctor-db-bad")
    (bad_home / "ragbase.db").write_bytes(b"garbage garbage garbage" * 100)
    check = _db_check(bad_home)
    assert check.level == doctor.FAIL and "数据库" in check.text, check

    # run_checks 里有“数据库”分组
    with env(RAG_HOME=str(home)):
        titles = [title for title, _ in doctor.run_checks(Settings(_env_file=None))]
    assert any("数据库" in title for title in titles), titles


def test_doctor_database_unwritable_dir_fails():
    home = _fresh_dir("doctor-db-ro")
    locked = home / "locked"
    locked.mkdir()
    os.chmod(locked, 0o555)
    try:
        if os.access(locked, os.W_OK):
            print("跳过只读目录的断言：当前用户（root？）无视目录权限")
            return
        check = _db_check(home, RAG_DB_PATH=str(locked / "ragbase.db"))
        assert check.level == doctor.FAIL and "数据库" in check.text and check.hint, check
    finally:
        os.chmod(locked, 0o755)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"通过：{test.__name__}")
    print(f"共 {len(tests)} 项测试通过")
