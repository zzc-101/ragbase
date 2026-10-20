# 内置 SQLite、会话持久化、体检进 Web：实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal：** 引入内置 SQLite，让对话会话在重启后完整保留，并把题集管理、体检（后台运行、报告历史、两次对比）做进 Web。

**Architecture：** 新增 `ragbase/db/` 负责连接、表结构版本和迁移；对话用 LangGraph 官方 SQLite checkpointer 外加我们自己的会话表和引用表；题集与体检报告各有一个存储类。CLI 与 HTTP API 共用这些存储和体检核心函数；Web 新增“体检”页。

**Tech Stack：** Python 标准库 `sqlite3`、`langgraph-checkpoint-sqlite==3.1.1`（兼容当前锁定的 `langgraph-checkpoint==4.1.1`；3.1.2 要求 ≥4.3.0，不能用）、FastAPI、LangChain 1.3.x 中间件、React + TypeScript + shadcn/ui。

**Spec：** [docs/superpowers/specs/2026-10-20-sqlite-persistence-design.md](../specs/2026-10-20-sqlite-persistence-design.md)（实施前必读，本计划只记录实现层面的决定）

## Global Constraints

- 数据库文件默认 `RAG_HOME/ragbase.db`，配置项 `RAG_DB_PATH` 可覆盖。
- 只用 `sqlite3` 和 `langgraph-checkpoint-sqlite==3.1.1`，不引入 ORM；新依赖写进 `pyproject.toml`（含 `constraint-dependencies` 里补锁的传递依赖）并 `uv lock`。
- 中文注释、中文文档、中文接口错误信息，风格与周边代码一致。
- 测试是 `tests/test_*.py` 纯 assert 脚本，末尾打印 `共 N 项测试通过`；用 `uv run python tests/run_all.py <名称片段…>` 只跑相关脚本，**不跑全量**。
- 测试只用临时目录里的数据库文件和假模型：不加载真实模型、不调用大模型、不碰真实索引、不读 `.env`、不运行 `docker compose config`。
- **不 git commit / push**（提交由用户决定）。
- 不改变现有接口的请求与响应格式（`/chat`、`/chat/stream`、`/chat/sessions*`、`/health` 等）。

## Review Focus

1. 线程池里多个会话同时发消息、同时有体检在写报告：不能出现 `database is locked` 报错（`busy_timeout` + 写锁）——Task 1 的并发写测试。
2. 没有 `chunk_id` 的片段在重启后要拿到同一个引用编号（`hash()` 每进程不同）——Task 2 的稳定键测试。
3. 导入的 JSON 不是数组、某题缺 `question`、字段类型不对：整体不写入，返回逐题错误——Task 3 的校验测试与 Task 4 的 422 测试。
4. 体检运行中有人编辑或删除题目：本次体检用开始时的题目快照，不受影响——Task 4 的测试。
5. 全新目录首次启动、旧版本库升级：自动建库 / 迁移，无需人工操作；目录不可写时 `doctor` 报 ✗、`serve` 给出中文错误——Task 1 的测试。

---

### Task 1：数据库基础

**Files：**
- Create：`src/ragbase/db/__init__.py`、`src/ragbase/db/core.py`、`src/ragbase/db/migrations.py`
- Modify：`src/ragbase/config.py`（`db_path` 设置与 `db_file()`）、`src/ragbase/doctor.py`（数据库检查）、`pyproject.toml`、`uv.lock`、`.gitignore`（`ragbase.db`、`ragbase.db-wal`、`ragbase.db-shm`）、`resources/env.example` 与 `.env.example`（加注释掉的 `RAG_DB_PATH`）
- Test：`tests/test_db.py`

**Interfaces：**
- Produces：
  - `ragbase.config.Settings.db_path: Path | None`（`RAG_DB_PATH`，空值视为未设置）；`ragbase.config.db_file(settings: Settings | None = None) -> Path`，未设置时为 `project_home() / "ragbase.db"`。
  - `class Database`（`db/core.py`）：
    - `Database(path: Path | str)`：`":memory:"` 表示内存库（供命令行 chat 和测试用，不开 WAL）。
    - `.path`
    - `query(sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]`
    - `transaction() -> ContextManager[sqlite3.Connection]`：进程内写锁 + `BEGIN IMMEDIATE`，异常时回滚。
    - `meta_get(key: str) -> str | None`
    - `meta_set(conn: sqlite3.Connection, key: str, value: str) -> None`：在调用方的事务里写。
    - `schema_version() -> int`
    - `new_connection() -> sqlite3.Connection`：同一文件的另一条连接，给 LangGraph checkpointer 用。
    - `close() -> None`
  - 连接参数：`check_same_thread=False`，`row_factory=sqlite3.Row`，`PRAGMA journal_mode=WAL`（文件库）、`foreign_keys=ON`、`busy_timeout=5000`。
  - `db/migrations.py`：`MIGRATIONS: list[Callable[[sqlite3.Connection], None]]`，第 i 个迁移把版本升到 i+1，版本存在 `PRAGMA user_version`；`migrate(conn) -> int`，每个迁移各自在一个事务里执行。迁移 1 建 `meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)`。
  - `ragbase.db.open_database(settings: Settings | None = None) -> Database`：创建父目录、打开、迁移到最新。目录不可写或文件损坏时抛 `DatabaseError(中文说明)`（`ragbase.db.DatabaseError`）。后续任务会在这里追加“打开后钩子”（Task 3 的首次导入）。

- [ ] **Step 1：写失败测试** `tests/test_db.py`
  - `test_fresh_file_is_created_and_migrated`：在临时目录里 `open_database` → 文件存在，`schema_version() == len(MIGRATIONS)`，`meta` 表可读写。
  - `test_reopen_is_idempotent`：同一文件打开两次，版本不变、数据还在。
  - `test_upgrade_from_older_version`：手工建一个 `user_version=0` 的空库 → 打开后升到最新。
  - `test_transaction_rolls_back_on_error`：事务里写入后抛异常 → 数据不存在。
  - `test_concurrent_writers_do_not_raise_locked`：8 个线程各在 `transaction()` 里写 50 行，同时另一个线程用 `new_connection()` 在自己的事务里写 → 无异常，行数正确。
  - `test_memory_database`：`Database(":memory:")` 迁移成功。
  - `test_unwritable_dir_gives_chinese_error`：`RAG_DB_PATH` 指向只读目录下的文件 → `DatabaseError`，消息含“数据库”。
  - `test_db_path_setting`：未设置时 `db_file()` 等于 `project_home()/"ragbase.db"`；设为空串时视为未设置。
- [ ] **Step 2：** `uv run python tests/run_all.py test_db` → 失败（模块不存在）。
- [ ] **Step 3：** 实现上面的接口；`pyproject.toml` 加 `langgraph-checkpoint-sqlite==3.1.1` 并 `uv lock`（确认 `uv.lock` 里 `langgraph-checkpoint` 仍是 4.1.1）；`doctor` 增加一项：能打开数据库、版本是最新（✓ / ✗ 加中文说明）。
- [ ] **Step 4：** `uv run python tests/run_all.py test_db config doctor cli` → 全部通过。

### Task 2：对话会话持久化

**Files：**
- Create：`src/ragbase/agent/store.py`
- Modify：`src/ragbase/agent/service.py`、`src/ragbase/agent/tools.py`（`CitationRegistry`）、`src/ragbase/db/migrations.py`（迁移 2）、`src/ragbase/config.py`（废弃两项会话配置）、`src/ragbase/api/app.py`（启动时打开数据库、会话接口）、`src/ragbase/cli.py`（`chat` 用内存库）
- Test：新建 `tests/test_chat_persistence.py`；修改 `tests/test_agent.py`（去掉与淘汰相关的用例，裁剪用例改成“状态完整、模型只看到最近 N 轮”）

**Interfaces：**
- Consumes：Task 1 的 `Database`、`open_database`、`MIGRATIONS`。
- Produces：
  - 迁移 2：
    - `chat_sessions(session_id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL, message_count INTEGER NOT NULL, committed_citations INTEGER NOT NULL DEFAULT 0)`
    - `chat_citations(session_id TEXT NOT NULL, number INTEGER NOT NULL, chunk_key TEXT NOT NULL, document TEXT NOT NULL, PRIMARY KEY(session_id, number), UNIQUE(session_id, chunk_key))`
    - `document` 为 JSON：`{"page_content": …, "metadata": {…}}`。
  - `class ChatStore(db: Database)`（`agent/store.py`）：
    - `record_turn(session_id: str, first_message: str, added_messages: int, committed_citations: int, now: float) -> None`：首轮插入（标题取首条用户消息，压成单行，截到 `TITLE_MAX_CHARS`）；之后累加 `message_count`、更新 `updated_at` 和 `committed_citations`。
    - `list_sessions() -> list[dict]`：返回 `{session_id, title, updated_at, message_count}`，按 `updated_at` 倒序，格式与现有 `/chat/sessions` 相同。
    - `exists(session_id) -> bool`
    - `committed_citations(session_id) -> int`（会话不存在时为 0）
    - `load_citations(session_id) -> list[tuple[int, str, Document]]`
    - `add_citation(session_id, number, chunk_key, doc) -> None`
    - `truncate_citations(session_id, keep: int) -> None`：删除 `number > keep` 的行。
    - `delete_session(session_id) -> bool`
  - `CitationRegistry`：
    - `CitationRegistry(store: ChatStore | None = None, session_id: str | None = None)`：有 store 时从库里加载已有编号，新登记的写穿到库里；`rollback(mark)` 同时调用 `truncate_citations`。
    - `_key` 对没有 `chunk_id` 的片段改用 `md5(source|page|page_content)`。
  - `build_chat_service(pipeline, settings, model=None, db: Database | None = None) -> ChatService | None`：
    - `db` 为 `None` 时用 `Database(":memory:")` 加 `InMemorySaver`；
    - 否则用 `SqliteSaver(db.new_connection())`（调用 `setup()`）加 `ChatStore(db)`。
    - 两种模式走同一套代码。
  - `ChatService(agent, checkpointer, settings, store: ChatStore, clock=time.time)`：
    - `SessionRegistry` 只保留“每会话一把锁 + 内存中的 `CitationRegistry` 缓存”，没有 TTL / 数量上限。
    - `list_sessions()` 直接查 `store.list_sessions()`。
    - `delete(session_id)` 先删 checkpointer 的 thread，再在一个事务里删会话行和引用行。两步不在同一事务里（我的决定：checkpointer 用独立连接）；中途失败时，再删一次可以继续删完。
    - `history()` 返回全部消息。
  - 历史窗口：把 `HistoryTrimMiddleware` 改成用 `wrap_model_call` / `awrap_model_call` 截取发给模型的消息（`request.override(messages=…)`，LangChain 1.3.14 已有），不再写 `RemoveMessage`。截取规则仍是整轮保留或丢弃、保留最近 `chat_history_turns` 轮（包括本轮）。
  - 半截轮次恢复：每轮开始（拿到会话锁之后）检查状态末尾。有 `tool_calls` 没对上 `ToolMessage`，或者最后一条用户消息之后没有不带 `tool_calls` 的 AI 回答，就把最后一条用户消息起的整段删掉（`update_state` + `RemoveMessage`），并且 `truncate_citations(session_id, store.committed_citations(session_id))`。
  - checkpoint 瘦身：每轮成功后，删掉这个 thread 除最新一个之外的 checkpoint 及其 writes。先查 3.1.1 的 `SqliteSaver` 是否有 `prune` / 等价方法，没有就直接对它的 `checkpoints`、`writes` 表执行 SQL。
  - 配置：`chat_max_sessions`、`chat_session_ttl` 仍能解析；设置了的话，在 `build_chat_service` 里记一条 warning“已废弃、不再生效”；从 `env.example` 删掉。
  - API：
    - lifespan 里 `app.state.db = open_database(settings)`，`app.state.chat_store = ChatStore(app.state.db)`，`chat_factory` 收到 `db`。
    - `app.state.chat is None` 时，`GET /chat/sessions` 从 `chat_store` 读；`GET /chat/sessions/{id}` 用一个只读的 `SqliteSaver` 读消息并照常返回；`DELETE` 照常可用。发消息仍回 503。
    - 应用关闭时 `db.close()`。
  - CLI：`ragbase chat` 调 `build_chat_service(..., db=None)`。

- [ ] **Step 1：写失败测试** `tests/test_chat_persistence.py`（脚本化假模型的写法照 `tests/test_agent.py`）：
  - `test_restart_keeps_sessions_history_and_citations`：用文件库构造 ChatService，跑 3 轮带引用的对话 → 关闭 → 用同一文件再构造一个 → `list_sessions` 一致，`history()` 消息与引用一致；第 4 轮引用同一片段时编号不变，新片段的编号接着之前的最大编号。
  - `test_full_history_kept_but_model_sees_window`：`chat_history_turns=2` 时跑 4 轮 → `history()` 里有 4 条用户消息；假模型在第 4 轮收到的消息里只有最近 2 条用户消息。
  - `test_title_fixed_and_count_cumulative`：第一轮之后标题不变；`message_count` 等于 用户消息数 + 最终回答数。
  - `test_failed_turn_rollback_persists`：一轮失败后，库里没有该轮新登记的引用行，`message_count` 不变。
  - `test_recover_incomplete_turn_after_crash`：用 `update_state` 人为写入“用户消息 + 带 tool_calls 的 AI 消息”，再插入一行多余的引用 → 下一轮正常完成，残缺的那轮和多余的引用都被清掉。
  - `test_citation_key_stable_without_chunk_id`：没有 `chunk_id` 的片段，在两个进程里算出的 key 相同（用子进程算一次，和当前进程比较）。
  - `test_delete_removes_everything_and_survives_restart`：删除后重开，会话列表里没有它，checkpointer 里也没有这个 thread。
  - `test_old_checkpoints_pruned`：跑 3 轮后，checkpointer 表里这个 thread 只剩 1 个 checkpoint。
  - `test_sessions_endpoints_when_agent_not_ready`：`create_app` 的 `chat_factory` 返回 `None`，库里预先有会话 → `GET /chat/sessions` 能列出，`GET /chat/sessions/{id}` 返回消息，`DELETE` 返回成功，`POST /chat` 返回 503。
  - `test_deprecated_session_settings_warn`：设置 `RAG_CHAT_SESSION_TTL` 后能启动，并有 warning。
- [ ] **Step 2：** `uv run python tests/run_all.py chat_persistence` → 失败。
- [ ] **Step 3：** 按上面的 Interfaces 实现；同步更新 `test_agent.py`：删掉 TTL / LRU 和淘汰相关的两个用例，`test_history_is_bounded_by_turns` 改成断言“模型输入有界、状态完整”；更新 `agent/service.py` 和 `api/app.py` 的模块说明。
- [ ] **Step 4：** `uv run python tests/run_all.py chat_persistence agent api auth db config` → 全部通过。

### Task 3：题集与报告入库，CLI 改为读写数据库

**Files：**
- Create：`src/ragbase/evaluation/questions.py`、`src/ragbase/diagnostics_reports.py`
- Modify：`src/ragbase/db/migrations.py`（迁移 3）、`src/ragbase/db/core.py`（`open_database` 打开后执行首次导入）、`src/ragbase/diagnostics.py`、`src/ragbase/evaluation/runner.py`、`src/ragbase/cli.py`
- Test：新建 `tests/test_questions.py`；修改 `tests/test_diagnose.py`

**Interfaces：**
- Consumes：Task 1 的 `Database` / `open_database` / `meta_get` / `meta_set`。
- Produces：
  - 迁移 3：
    - `eval_questions(id INTEGER PRIMARY KEY, question TEXT NOT NULL, reference TEXT, difficulty TEXT, type TEXT, source_topic TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)`
    - `diagnose_reports(id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, origin TEXT NOT NULL CHECK(origin IN ('cli','web')), mode TEXT NOT NULL, question_count INTEGER NOT NULL, params TEXT NOT NULL, summary TEXT NOT NULL, recommendations TEXT NOT NULL, records TEXT NOT NULL, extra TEXT NOT NULL)`
    - 各 JSON 列存 `build_report` 产出的对应部分；`extra` 存 `question_ids`、`question_digest`、`concurrent_chat_requests`、`warmup_question`。
  - `ragbase.evaluation.questions`：
    - `QuestionValidationError(errors: list[dict])`，每项 `{index, field, message}`，`message` 用中文。
    - `class QuestionStore(db)`：
      - `list() -> list[dict]`
      - `get(id) -> dict | None`
      - `create(data: dict) -> dict`
      - `update(id, data) -> dict | None`
      - `delete(id) -> bool`
      - `import_items(items: object, mode: Literal["append","replace"]) -> dict`：返回 `{"imported": int, "reassigned": [[old, new], …]}`；校验失败抛 `QuestionValidationError`，不写入任何数据。
      - `export() -> list[dict]`
    - 题目字段：`id, question, reference, difficulty, type, source_topic, created_at, updated_at`。
    - 导入时 `reference = item.get("reference") or item.get("answer_keyword")`。导出时同时写出 `reference` 和 `answer_keyword`（取值相同）。
    - `ensure_initial_import(db, json_path: Path) -> int | None`：没有 `meta["eval_questions_imported"]` 标记、表为空、文件存在时导入并写标记，返回导入条数；否则返回 None。
  - `ragbase.diagnostics_reports.ReportStore(db)`：
    - `save(report: dict, origin: str) -> int`
    - `list() -> list[dict]`：每项 `{id, created_at, origin, mode, question_count, candidates_recall, reranked_recall, refusal_rate, total_ms_mean, rerank_share_pct, concurrent_chat_requests}`，按 id 倒序。
    - `get(id) -> dict | None`：结构与 `build_report` 相同，外加 `id`、`origin`。
    - `delete(id) -> bool`
  - `ragbase.diagnostics` 改动：
    - `measure_question(..., mode: str | None = None)`：把 mode 传给 `retrieve_and_rerank`。
    - `reference_of` 改读 `reference` 字段。
    - `load_questions(store: QuestionStore, limit) -> tuple[list[dict], int]`：取代读文件的版本；题集为空时抛 `SystemExit`，提示“在 Web 的体检页添加题目，或 ragbase questions import”。
    - 新增核心函数 `run_diagnosis(pipeline, settings, questions: list[dict], mode: str, load_seconds: float, duplicates_skipped: int, on_progress: Callable[[int, int], None] | None = None, chat_requests: Callable[[], int] | None = None) -> dict`：先预热，再逐题测量，返回 `build_report(...)` 的结果。其中 `extra.concurrent_chat_requests` = 结束时 `chat_requests()` − 开始时的值（`chat_requests` 为 None 时记 0）。
    - `build_report` 增加 `question_ids`、`question_digest`（按 id 排序后对 `id|question|reference` 做 md5）、`concurrent_chat_requests`。
    - CLI 的 `run(args)` 调 `run_diagnosis`，结果存进 `ReportStore`（origin=`cli`），打印报告和报告 id。`--json PATH` 可选导出；默认不再写 `diagnose_report.json`。
  - `ragbase.evaluation.runner.load_eval_questions(start_id, limit)` 改为从 `QuestionStore` 读；`reference` 为空的题跳过并打印提示。
  - CLI：
    - `ragbase questions import <file> [--replace]`：打印导入条数和换了 id 的题；校验失败时逐条打印错误，退出码非零。
    - `ragbase questions export <file>`
    - `ragbase diagnose --json PATH`
    - 同步更新 `cli.py` 模块说明里的子命令清单。

- [ ] **Step 1：写失败测试** `tests/test_questions.py`：
  - `test_crud`：增改删查。
  - `test_import_validation_is_atomic`：传入非数组、某题缺 `question`、`id` 不是整数 → 抛 `QuestionValidationError`，`errors` 里带 index / field，表保持原样。
  - `test_import_append_reassigns_conflicting_ids`
  - `test_import_replace_keeps_file_ids`
  - `test_reference_falls_back_to_answer_keyword`
  - `test_export_is_backward_compatible`：导出结果交给旧的 `reference or answer_keyword` 逻辑能读出同样的参考答案。
  - `test_initial_import_runs_once`：首次导入；删光题目后再次 `open_database` 不会重新导入。
  - `test_report_store_roundtrip_and_list_fields`

  修改 `tests/test_diagnose.py`：
  - 题目改从临时库读。
  - `run_diagnosis` 调用 `on_progress` 的次数等于题数。
  - `concurrent_chat_requests` 取的是差值。
  - `question_digest` 与题目顺序无关、与内容有关。
  - CLI 不再默认写 `diagnose_report.json`，`--json` 时才写。
- [ ] **Step 2：** `uv run python tests/run_all.py questions diagnose` → 失败。
- [ ] **Step 3：** 实现。`evaluation/runner.py` 仍会在顶层导入 ragas，`diagnostics.py` 不能导入它；`QuestionStore` 放在轻量模块里。
- [ ] **Step 4：** `uv run python tests/run_all.py questions diagnose db fixes cli docs` → 全部通过。

### Task 4：题集、体检、报告的 HTTP API

**Files：**
- Create：`src/ragbase/diagnostics_jobs.py`
- Modify：`src/ragbase/ingest/jobs.py`（抽出共用的“忙”锁）、`src/ragbase/api/app.py`、`src/ragbase/api/schemas.py`、`src/ragbase/api/auth.py`（公开 SPA 路径 `/diagnose`）
- Test：新建 `tests/test_api_diagnose.py`；`tests/test_auth.py` 补充用例

**Interfaces：**
- Consumes：Task 3 的 `QuestionStore`、`ReportStore`、`run_diagnosis`、`QuestionValidationError`；Task 2 中 `app.state.db` 的打开方式。
- Produces：
  - `ingest/jobs.py`：
    - `class BusyLock`：
      - `acquire(holder_id: str, kind: str) -> None`：拿不到时抛 `JobRunningError(running_id, kind)`。
      - `release() -> None`
      - 属性：`running_id`、`running_kind`
    - `IngestJobManager(busy: BusyLock | None = None)`。
    - `JobRunningError` 的提示改为按 kind 区分：“已有入库任务在运行” / “已有体检在运行”。
  - `diagnostics_jobs.py`：
    - `DiagnoseJob` dataclass：`job_id, status, total, done, report_id, error, started_at, finished_at`，带 `to_dict()`。
    - `DiagnoseJobManager(busy: BusyLock)`：`start(run: Callable[[DiagnoseJob], int]) -> DiagnoseJob`（`run` 返回报告 id），`get(job_id) -> DiagnoseJob | None`。
  - `app.state.jobs` 与 `app.state.diagnose_jobs` 共用一个 `BusyLock`；新增 `app.state.chat_requests` 计数器，`/chat`、`/chat/stream` 每次请求加一（线程安全）。lifespan 里记录模型加载耗时：`app.state.model_load_seconds`。
  - 接口（全部需要 API Key；schemas 进 `api/schemas.py`）：

    | 方法与路径 | 成功 | 错误 |
    |---|---|---|
    | `GET /eval-questions` | 200 `{items: [...]}` | |
    | `POST /eval-questions` | 201 返回该题 | 422 |
    | `PUT /eval-questions/{id}` | 200 | 404 / 422 |
    | `DELETE /eval-questions/{id}` | 200 `{deleted: true}` | 404 |
    | `POST /eval-questions/import`，body `{items: [...], mode: "append"\|"replace"}` | 200 `{imported, reassigned}` | 422 `{detail, errors: [{index, field, message}]}` |
    | `GET /eval-questions/export` | 200，JSON 数组，带 `Content-Disposition: attachment; filename="eval_questions.json"` | |
    | `POST /diagnose`，body `{mode?: "vector"\|"bm25"\|"hybrid", limit?: int ≥1}` | 202，返回 `DiagnoseJob` | 400 题集为空；409 有任务在跑；503 流水线未就绪或索引为空 |
    | `GET /diagnose/jobs/{job_id}` | 200 | 404 |
    | `GET /diagnose/reports` | 200 `{items: [...]}`，字段同 `ReportStore.list()` | |
    | `GET /diagnose/reports/{id}` | 200 完整报告 | 404 |
    | `DELETE /diagnose/reports/{id}` | 200 | 404 |

  - `POST /diagnose` 在返回 202 之前就取好题目快照（先 `store.list()`，再按 limit 截取并去重），后台线程调用 `run_diagnosis(app.state.pipeline, settings, questions, mode or settings.mode, app.state.model_load_seconds, dup, on_progress=…, chat_requests=…)`，用 `ReportStore.save(report, "web")` 保存。

- [ ] **Step 1：写失败测试** `tests/test_api_diagnose.py`（`create_app` 用假流水线，写法照 `tests/test_api.py`）：
  - 题集 CRUD 的状态码。
  - 导入非法内容返回 422 并带 errors，且表没有变化。
  - 导出带 attachment 头。
  - `POST /diagnose` → 轮询到 succeeded → `GET /diagnose/reports` 里出现这份报告，origin 为 web。
  - 题集为空时返回 400。
  - 入库任务进行中返回 409；反过来，体检进行中发起 `POST /ingest` 也返回 409。
  - 体检进行中删除一道题，报告的 `question_count` 仍是开始时的数量。
  - 删除报告后再查返回 404。
  - 设置 `RAG_API_KEY` 时，这些接口不带密钥返回 401，`GET /diagnose`（SPA 路由）公开可访问。
- [ ] **Step 2：** `uv run python tests/run_all.py api_diagnose` → 失败。
- [ ] **Step 3：** 实现。
- [ ] **Step 4：** `uv run python tests/run_all.py api_diagnose api auth ingest_service agent chat_persistence` → 全部通过。

### Task 5：Web “体检”页

**Files：**
- Create：
  - `web/src/pages/DiagnosePage.tsx`
  - `web/src/components/diagnose/`：`RunPanel.tsx`、`ReportView.tsx`、`ReportHistory.tsx`、`ReportCompare.tsx`、`QuestionTable.tsx`、`QuestionDialog.tsx`
  - `web/src/hooks/useDiagnoseJob.ts`
- Modify：`web/src/App.tsx`（路由 `diagnose`）、`web/src/components/AppShell.tsx`（导航加“体检”）、`web/src/lib/api.ts`（类型与函数）、`web/src/hooks/queries.ts`（`useQuestions`、`useReports`、`useReport(id)`）

**Interfaces：**
- Consumes：Task 4 的接口与字段（`api.ts` 里的类型和 `api/schemas.py` 一一对应）。
- Produces：页面行为与 spec 5.6 一致。具体要求如下：
  - 进度轮询照 `useIngestJob` 的写法；页面刷新后可以通过 `GET /diagnose/jobs/{id}` 恢复，job id 存在 sessionStorage，读写一律经过 `safeStorage`。
  - 报告区默认显示最新一份报告；在历史列表里点击某一份则切换显示它。
  - 历史列表可勾选恰好两份报告进入对比：
    - 参数不同的项高亮；
    - 质量和速度显示差值，质量升高和耗时降低用成功色，反之用警示色；
    - `question_digest` 不同时提示“题集不同，质量指标不可直接比较”；
    - 来源不同、或任一份 `concurrent_chat_requests > 0` 时也给出提示。
  - 各阶段耗时占比用 Tailwind 画横向条，不引入图表库。
  - 删除报告、删除题目、以“替换”方式导入，都要先弹 AlertDialog 确认。
  - 导入用文件选择框读取本地 JSON；导出走 `GET /eval-questions/export` 再触发下载（要带上鉴权头，所以用 `fetch` 拿到 blob 后下载，不能直接用链接）。
  - 窄屏（< 768px）下各区块纵向排列，不出现横向滚动。

- [ ] **Step 1：** 加 `api.ts` 的类型、函数和 hooks；然后 `pnpm -C web build` 应该通过（页面还没接入）。
- [ ] **Step 2：** 实现页面、组件、路由和导航。
- [ ] **Step 3：** `pnpm -C web lint` 和 `pnpm -C web build` 都通过。
- [ ] **Step 4（主会话验收时执行）：** 在浏览器面板里打开页面（用 `.claude/launch.json` 的 `rag-service` 启动本地服务，只加载本地模型，不调用大模型）；跑一次 `limit=3` 的体检，检查进度、报告、历史、对比、题集增删改和导入导出，再把宽度切到 375px 检查一遍。

### Task 6：文档与收尾

**Files：**
- Modify：
  - `docs/diagnostics.md`、`docs/evaluation.md`、`docs/api.md`、`docs/architecture.md`、`docs/deployment.md`
  - `docs/configuration.md`：配置区块用 `uv run python scripts/gen_config_docs.py` 重新生成
  - `scripts/gen_config_docs.py`：如有需要，补 `RAG_DB_PATH` 的说明，并标注两个已废弃的配置
  - `README.md`、`CHANGELOG.md`

- [ ] **Step 1：** 按 spec 5.7 和 Task 1–5 的实际接口更新文档。`deployment.md` 写清楚备份方式：停服务后拷贝，或用 `sqlite3 ragbase.db ".backup 目标文件"`；并注明 Docker Desktop（macOS）绑定目录上使用 WAL 的风险。
- [ ] **Step 2：** `uv run python tests/run_all.py docs config cli` → 全部通过。
- [ ] **Step 3：** 汇总各任务跑过的测试；全量测试留给用户和主会话一起跑。
