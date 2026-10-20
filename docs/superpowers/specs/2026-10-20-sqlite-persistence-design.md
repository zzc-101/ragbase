# 设计：内置 SQLite、会话持久化、体检进 Web（第一期）

日期：2026-10-20　状态：待用户确认（审阅意见已处理）

## 1. 背景与目的

ragbase 的主线从“一组脚本”转向“一套框架”：**Web 是核心入口，负责可视化管理**，命令行只做辅助、后续逐步弱化。

目前有两个直接的痛点：

1. **服务重启后对话全部丢失**：会话历史存在 LangGraph 的 `InMemorySaver`，会话级引用编号（`CitationRegistry`）也只在内存里。
2. **体检报告没有历史可对比**：刚做好的 `ragbase diagnose` 只有命令行，每次覆盖写 `diagnose_report.json`；题集 `eval_questions.json` 只能在服务器上手改文件，只用 Web 的人无法提供。

本期引入一个内置 SQLite 数据库，解决这两个痛点，并把体检（含题集管理、报告历史与对比）做进 Web。

## 2. 目标 / 非目标 / 反目标

**目标**

- 内置 SQLite（零额外服务、开箱即用），有表结构版本与自动迁移，后续各期可以往里加表。
- 对话会话持久化：重启后会话列表、每轮问答、引用编号与引用片段都还在；追问时编号继续接着之前的编号走。
- 会话只在用户手动删除时才删除，不再按数量 / 空闲时间自动清理。
- 题集存进数据库，Web 上可增删改、导入 / 导出 JSON。
- 体检可以从 Web 发起（后台任务 + 进度），每次结果作为一条历史报告保存；Web 上可查看任意一次报告、删除报告、选两次报告并排对比。
- 命令行 `ragbase diagnose` / `ragbase eval` 与 Web 共用同一份题集和同一张报告表。

**非目标（本期不做）**

- Web 上修改配置（仍用 `.env`）。
- 入库任务记录、评测（`ragbase eval`）结果搬进数据库。
- 多个题集；报告的导出 / 分享；用户 / 权限体系（ragbase 只提供 RAG 能力，不做用户管理）。
- 支持 PostgreSQL 等其他数据库。

**反目标（做到了也算失败）**

- 重启后追问时引用编号错乱（例如新回答的 [1] 指向了别的片段）——这是之前修过的 bug，持久化不能把它带回来。
- 体检运行期间对话完全不可用或长时间卡死。
- 数据库文件写坏 / 半写导致服务起不来。
- 需要用户手动建库、手动跑迁移才能用。

## 3. 约束

- 单进程服务（`ragbase serve` 不支持多 worker），FastAPI 线程池里多线程访问数据库。
- 只用 Python 标准库 `sqlite3` + LangGraph 官方 SQLite checkpointer（`langgraph-checkpoint-sqlite`，选与当前 `langgraph==1.2.10` / `langgraph-checkpoint==4.1.1` 兼容的版本，写进 `pyproject.toml` 并更新 `uv.lock`）；不引入 ORM。
- 所有默认路径跟随 `RAG_HOME`。
- 现有约定不变：中文注释 / 文档；测试是 `tests/` 下的纯 assert 脚本（`tests/run_all.py` 汇总）；前端 React + Vite + TypeScript + Tailwind + shadcn/ui，保持现有“安静阅读器”风格和响应式布局。
- 鉴权：静态 API Key 中间件是“默认拒绝 + 白名单”。新增的 API 都需要密钥；新增的前端路由（SPA 页面路径）要加进 `api/auth.py` 的公开路径。

## 4. 已确定的决定（来自用户）

- 引入内置数据库，选 SQLite。
- 本期解决的痛点：重启后对话丢失、报告没有历史可对比。
- 会话保留策略：**只在手动删除时删除**。
- 长会话**完整保存**全部消息（Web 上能翻看整段对话），只在发给大模型时截取最近 `RAG_CHAT_HISTORY_TURNS` 轮。
- Web 要支持题集管理。
- 体检页面不需要先看草图，功能完整即可。

## 5. 设计

### 5.1 数据库基础

- 文件：`RAG_HOME/ragbase.db`，可用新配置项 `RAG_DB_PATH` 覆盖（我的决定）。加入 `.gitignore`；`docs/deployment.md` 的备份一节补上它。
- 新模块 `src/ragbase/db/`：负责打开连接（`check_same_thread=False`，开启 WAL、`foreign_keys=ON`、合理的 `busy_timeout`）、写操作串行（一把进程内锁即可）、表结构版本与迁移。
- 版本与迁移：`PRAGMA user_version` 记录版本；迁移是代码里按版本号排列的一组 SQL / 函数，启动时自动执行到最新；每个迁移在事务里完成。
- 何时打开：`ragbase serve` 启动时；命令行里只有用到它的子命令（`diagnose`、`eval`、`questions`）才打开。命令行 `ragbase chat` 仍用内存保存会话、不写库（我的决定：命令行是辅助入口，临时会话不该混进 Web 的会话列表，也避免两个进程同时写对话表）；`build_chat_service` 因此要同时支持内存模式和数据库模式。`ragbase doctor` 增加一项检查：数据库可打开、版本是最新。
- LangGraph checkpointer 的表由它自己的 `setup()` 管理，与我们自己的表放在同一个文件里（我的决定：备份只有一个文件）。

### 5.2 对话会话持久化

现状（`src/ragbase/agent/service.py`）：

- 会话 = LangGraph thread，历史在 `InMemorySaver`。
- `SessionRegistry` 管理会话的最近使用时间、每会话一把锁和一个 `CitationRegistry`，并按 `RAG_CHAT_MAX_SESSIONS`（200）/ `RAG_CHAT_SESSION_TTL`（3600 秒）淘汰。
- `list_sessions` 从内存里的会话列表 + checkpointer 拼出标题、更新时间、消息数。

改为：

- checkpointer 换成 SQLite 版（同一数据库文件）。
- 新表 `chat_sessions`：`session_id`（主键）、`title`、`created_at`、`updated_at`、`message_count`。每轮成功结束后更新；`list_sessions` 直接查这张表，按 `updated_at` 倒序。
- 新表 `chat_citations`：`session_id`、`number`、`chunk_key`、`document`（JSON：正文 + 元数据，含精排得分等）；主键 (`session_id`, `number`)，(`session_id`, `chunk_key`) 唯一。
- `CitationRegistry` 改为由数据库支撑（或在内存里缓存、写穿到数据库）：
  - 编号规则不变：同一片段同一编号，跨轮连续，重启后接着编。
  - 失败回滚（`mark` / `rollback`）要同时撤销数据库里本轮新登记的行。
  - **注意**：`_key` 对没有 `chunk_id` 的片段用的是 Python 内置 `hash()`，它每个进程都不一样；持久化后必须改成稳定的哈希（如 md5），否则重启后同一片段会被当成新片段。
- 删除：`DELETE /chat/sessions/{id}`（Web 上的删除）删掉该会话的 checkpointer 数据、`chat_citations`、`chat_sessions` 行，在一个事务里完成。
- 取消自动淘汰：去掉 TTL / 数量上限淘汰。`RAG_CHAT_MAX_SESSIONS`、`RAG_CHAT_SESSION_TTL` 废弃：仍能被解析（避免旧 `.env` 启动报错），设置了就在启动日志里给一条“已废弃、不再生效”的警告；从文档和 `.env.example` 中移除（我的决定）。
- 每会话一把锁（防同一会话并发两轮）保留，可以按需在内存里创建，不必持久化。
- 本期之前已存在于内存的会话不需要迁移（重启本来就会丢）。
- 半截轮次的恢复（我的决定，本期要做）：LangGraph 每个节点都会写 checkpoint，进程在一轮中途被杀（kill -9 / 断电）后，库里可能留下“用户消息 + 带 tool_calls 的 AI 消息、缺少对应工具结果”这种残缺状态，下一轮会被大模型接口拒绝；写穿的引用表也会留下本轮多登记的行。每轮开始时检测末尾不完整的一轮并整轮裁掉，同时删掉编号大于该会话最后一个完整轮次所引用最大编号的引用行（或用等价的“已提交编号”标记）。
- checkpoint 体积（我的决定）：每轮成功结束后只保留每个 thread 最新的 checkpoint，删掉旧的，避免无限保留会话时库无限膨胀。`/chat/sessions` 本期不分页。
- 对话服务未就绪（大模型未配置或索引为空，`app.state.chat is None`）时（我的决定）：会话列表和查看历史照常从库里读，删除照常可用；只有发消息返回 503。
- 完整保存历史（用户决定）：现在的 `HistoryTrimMiddleware` 在每轮开头用 `RemoveMessage(REMOVE_ALL_MESSAGES)` 把状态裁到最近 N 轮，旧消息被永久删掉。改为**状态里保留全部消息，只在调用大模型时截取最近 N 轮**（例如用中间件的模型调用包装钩子修改发给模型的消息列表，不回写状态）。截取规则不变：整轮保留或丢弃，不拆开 tool_calls 与工具结果。大模型看到的上下文长度与现在一致。
- 标题与计数（我的决定）：`title` 在会话第一轮成功时取第一条用户消息并固定，之后不变；`message_count` 为累计数（用户消息 + 最终回答，不含工具步骤），与现有口径一致。历史接口返回全部消息，引用表的行随会话保留（被截出模型上下文的旧轮次仍能在 Web 上看到引用）。
- 不改变现有的其他对话行为：工具调用、流式事件顺序、`/chat` / `/chat/stream` / `/chat/sessions*` 的请求与响应格式都保持不变。

### 5.3 题集

- 单个题集（本期不做多题集）。新表 `eval_questions`：`id`（整数主键）、`question`（必填）、`reference`（参考答案，可空：空时这题只计耗时、不计召回率）、`difficulty`、`type`、`source_topic`（可空）、`created_at`、`updated_at`。
- 字段约定（我的决定）：
  - 导入时 `reference` 取 JSON 里的 `reference`，没有则取 `answer_keyword`；不单独存 `answer_keyword`。
  - 导出时写 `reference`，并把同样的值写进 `answer_keyword`，保证导出的文件能被旧版 `ragbase eval` 直接读取。
  - `id`：导入时保留 JSON 里的 id；“追加”遇到已存在的 id 时改为自动分配新 id（结果里告诉用户哪些题换了 id）；“替换全部”保留文件里的 id。Web 上新增的题自动分配 id。`ragbase eval --start-id` 按这个 id 过滤，含义不变。
  - `ragbase eval` 遇到 `reference` 为空的题：跳过并打印提示（体检则照常计耗时）。
- 首次初始化：数据库里没有“已做过首次导入”标记、表为空且 `RAG_HOME/eval_questions.json` 存在时，自动导入一次，然后写入标记（存在一张小的键值 meta 表里）；任何打开数据库的命令都会触发这个检查（我的决定）。用户之后在 Web 上删光题目，重启也不会再次导入。之后数据库是唯一来源，JSON 只作为导入 / 导出格式。
- 导入：上传 JSON 数组（格式与现有 `eval_questions.json` 一致），选择“追加”或“替换全部”；校验失败时整体不导入，并返回哪一题、哪个字段有问题。
- 导出：下载同格式 JSON。
- 命令行：新增 `ragbase questions import <file> [--replace]` / `ragbase questions export <file>`；`ragbase diagnose` 和 `ragbase eval` 改为从数据库读题（数据库没有题时提示先导入，或从 Web 添加）。

### 5.4 体检报告历史

- 新表 `diagnose_reports`：`id`、`created_at`、`origin`（`web` / `cli`）、`mode`、`question_count`、`params`（JSON）、`summary`（JSON）、`recommendations`（JSON）、`records`（JSON，逐题记录）。内容就是现在 `build_report` 产出的结构。
- 保留到手动删除为止（与会话一致）。
- 命令行 `ragbase diagnose`：照旧在终端打印报告，结果写进数据库；不再默认写 `diagnose_report.json`，改为可选 `--json PATH` 导出（我的决定）。
- 服务内运行时，“模型加载耗时”取服务启动时实际加载模型的耗时（启动时记录下来）。服务内体检同样先做一次预热再计时（我的决定）。
- 报告里额外记录（我的决定）：运行期间服务处理的对话请求数（大于 0 时报告和对比页提示“耗时可能偏高”）；所用题目的 id 列表及其内容摘要（对比时若两份报告的题集不同就提示“题集不同，质量指标不可直接比较”）；来源（cli / web）不同时对比页也给出提示，但仍允许对比。

### 5.5 HTTP API

全部需要 API Key（与现有接口一致）。题集和报告接口不依赖对话 Agent，大模型未配置时照常可用；`POST /diagnose` 在索引为空时返回 503。请求 / 响应模型放进 `api/schemas.py`，出现在 `/docs`。

- 题集：`GET /eval-questions`、`POST /eval-questions`、`PUT /eval-questions/{id}`、`DELETE /eval-questions/{id}`、`POST /eval-questions/import`（JSON 数组 + `mode: append|replace`）、`GET /eval-questions/export`。
- 体检任务：`POST /diagnose`（可选 `mode`、`limit`）→ 202 返回任务；`GET /diagnose/jobs/{job_id}` → 状态、进度（已完成 / 总题数）、完成后的报告 id 或错误信息。
- 报告：`GET /diagnose/reports`（列表：id、时间、来源、模式、题数、几个关键数字）、`GET /diagnose/reports/{id}`（完整报告）、`DELETE /diagnose/reports/{id}`。
- 并发规则：
  - 同一时刻只运行一个体检；体检与入库 / 删除文档任务互斥（复用 `ingest/jobs.py` 那把“忙”锁或同等机制），冲突时回 409，与现有入库接口的行为一致。
  - 体检复用服务已加载的流水线；每题单独获取 `pipeline.lock`（现在的 `retrieve_and_rerank` 就是这样），所以对话请求能在题与题之间插进来——对话会变慢，但不会被整段阻塞。
  - 体检使用服务当前的召回参数；`mode` 可按次指定。
- 题集为空时 `POST /diagnose` 回 400 并说明原因。

### 5.6 Web

- 导航新增“体检”（路由 `/diagnose`），与“对话”“知识库”并列；`/diagnose` 加入鉴权的公开 SPA 路径。
- 页面包含：
  1. **运行**：召回方式选择、题数上限（可选）、“开始体检”按钮；运行中显示进度；提示“体检期间对话会变慢”。题集为空时按钮不可用，并引导去添加题目。
  2. **最新 / 当前查看的报告**：当前参数、质量（三组指标表、拒答率、Top1 得分分布）、速度（各阶段均值 / 中位数 / 最大 / 占比，占比用横向条形展示）、建议卡片（现象 / 可以尝试 / 可能的代价）。
  3. **历史**：报告列表（时间、来源、模式、题数、关键数字），可查看、删除（需确认），勾选两次报告进入对比。
  4. **对比**：两次报告并排，参数不同的项高亮，质量与速度给出差值（变好 / 变差用颜色区分）。
  5. **题集**：表格列出题目；新增 / 编辑（对话框）、删除（需确认）、导入（追加 / 替换，替换需确认）、导出。
- 布局在窄屏下可用（与现有页面一致）。

### 5.7 文档与清理

- 备份说明（我的决定）：WAL 模式下运行中只拷 `ragbase.db` 可能丢掉 `-wal` 里的数据，`docs/deployment.md` 写明停服务后拷贝，或用 `sqlite3 ragbase.db ".backup 目标文件"` 在线备份；并注明 Docker Desktop（macOS）绑定目录上使用 WAL 的已知风险（Docker 部署本身尚未验证）。
- 更新：`docs/diagnostics.md`（Web 用法、题集来源改为数据库、报告历史）、`docs/evaluation.md`（题集来源）、`docs/configuration.md`（`RAG_DB_PATH`、废弃两个会话配置；注意该文件部分区块由 `scripts/gen_config_docs.py` 生成，要用脚本重新生成）、`docs/api.md`（新接口）、`docs/architecture.md`（数据库与模块地图）、`docs/deployment.md`（备份 `ragbase.db`）、`README.md`、`CHANGELOG.md`。

## 6. 验收标准

- 发超过 `RAG_CHAT_HISTORY_TURNS` 轮带引用的对话 → 重启服务 → 会话列表、全部消息（包括超出窗口的早期轮次）、引用面板都还在，发给大模型的仍只有最近 N 轮；在该会话里追问，新引用编号接着之前的走，同一片段编号不变。
- 删除会话后，重启也不会再出现。
- 题集在 Web 上增删改、导入（追加 / 替换）、导出可用；旧的 `eval_questions.json` 在首次启动时被导入。
- Web 上发起体检能看到进度，完成后报告出现在历史里；可查看、删除、选两次对比。命令行 `ragbase diagnose` 的结果也出现在同一历史里（来源标为 cli）。
- 体检运行时可以正常发起对话（会变慢）；体检与入库任务同时发起时，后者回 409。
- 全新目录下首次启动自动建库；旧版本的数据库能自动迁移到新版本。

## 7. 测试

- 子代理按模块只做增量测试，用临时目录里的数据库文件和假模型；不加载真实模型、不调用大模型、不碰真实索引。
- 重点用例：
  - 用同一数据库文件构造两次 `ChatService`（模拟重启）→ 会话列表、历史、引用编号保持一致，失败回滚后数据库里没有残留。
  - 迁移从空库和从旧版本都能升到最新。
  - 题集导入的校验与原子性。
  - 体检任务与入库任务互斥。
  - 新接口的鉴权。
- 前端：`pnpm build` 与 `pnpm lint` 通过。
- 全量测试和真实模型下的端到端验证由用户和主会话一起做。

## 8. 留给实现者

- 表的具体列类型、索引、JSON 列的序列化细节。
- `CitationRegistry` 是“内存缓存 + 写穿”还是每次查库。
- 前端组件拆分、具体文案、对比视图的具体排版。
- 体检任务对象复用 `ingest/jobs.py` 的注册表还是另写一个轻量版本（满足 5.5 的互斥要求即可）。

## 9. 建议的实施顺序

1. 数据库基础模块（连接、迁移、配置项、doctor 检查）。
2. 会话持久化（checkpointer、会话表、引用表、去掉自动淘汰）。
3. 题集表 + 自动导入 + `ragbase questions` + `diagnose` / `eval` 改为读库 + 报告入库。
4. 题集 / 体检 / 报告 API。
5. Web “体检”页面。
6. 文档。

第 2 步与第 3 步只依赖第 1 步，互不依赖；第 4 步依赖第 3 步；第 5 步依赖第 4 步。
