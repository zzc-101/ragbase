"""内置 SQLite：连接 / 事务（core）与表结构迁移（migrations）。入口是 open_database()。"""
from ragbase.db.core import Database, DatabaseError, open_database
from ragbase.db.migrations import MIGRATIONS, migrate

__all__ = ["Database", "DatabaseError", "MIGRATIONS", "migrate", "open_database"]
