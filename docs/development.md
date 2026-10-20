# 开发指南

## 开发环境

需要 Python ≥ 3.12 和 [uv](https://docs.astral.sh/uv/)；前端另需 Node 22 与 pnpm。

```bash
git clone <本仓库地址> ragbase && cd ragbase
uv sync                       # 按 uv.lock 创建 .venv 并以可编辑模式安装 ragbase
uv run ragbase --help         # 命令行入口 ragbase = ragbase.cli:main（python -m ragbase 等价）
uv run ragbase init           # 生成 data/、.env（可选，开发时也可以只用环境变量）
```

- **在仓库根目录运行命令**（或设置 `RAG_HOME` 指向它）：数据、索引、`.env` 的默认位置都从项目主目录推导，见 [configuration.md](configuration.md#路径与项目主目录)。
- **依赖管理**：直接依赖在 `pyproject.toml` 里精确钉死；传递依赖的版本由 `[tool.uv] constraint-dependencies` 钉住（约束只限制版本，不额外引入依赖）。要升级某个传递依赖，先改这里再 `uv lock`。Linux 上 torch 走 PyTorch 官方 CPU 源（`[tool.uv.sources]`），macOS 用 PyPI 版以支持 MPS。
- 仓库根目录的 `README.md` 被 `pyproject.toml` 的 `readme` 引用，且 `Dockerfile` 会复制它——**不要删除或改名**。
- 日志：`RAG_LOG_LEVEL=DEBUG` 可看更多细节。

### 前端（web/）

技术栈：React 19 + Vite + TypeScript + Tailwind CSS 4 + shadcn/ui（radix），路由 react-router，数据请求 TanStack Query，回答用 react-markdown 渲染。

```bash
# 开发：后端在 8000 端口运行（uv run ragbase serve），前端 dev server 把 API 请求代理过去
cd web && pnpm install && pnpm dev          # http://localhost:5173，热更新
# 检查
cd web && pnpm build                        # tsc -b（严格类型检查）+ vite build，产物在 web/dist
cd web && pnpm lint                         # oxlint
# 生产：构建后由 FastAPI 在同一端口托管，不需要另起前端服务
scripts/build_web.sh                        # pnpm install --frozen-lockfile + build，并复制到 src/ragbase/web/dist（随包分发）
```

- dev server 的代理在 `web/vite.config.ts`：把 API 路径前缀（`/health`、`/search`、`/ask`、`/documents`、`/ingest`、`/chat` 等）转发到 `http://127.0.0.1:8000`。**新增了以其他前缀开头的 API 路由，要把前缀加进 `apiPaths`**（题集 `/eval-questions*` 和体检 `/diagnose*` 同理）；前端路由（`/`、`/library`、`/diagnose`、`/c/:id`）刷新页面时不能被代理走。注意 `/diagnose` 既是前端页面（`GET`）又是体检接口（`POST /diagnose`、`/diagnose/jobs/*`、`/diagnose/reports*`），所以 `vite.config.ts` 对它单独配置：浏览器的页面导航（`Accept` 含 `text/html`）不走代理，刷新页面才能拿到前端。
- API 服务托管 `web/dist`：目录是否存在在请求时检查，服务运行中重新 `pnpm build` 立即生效（细节见 [api.md](api.md#前端路由与静态资源)）。
- 目录结构：`src/pages/`（`ChatPage`、`LibraryPage`、`DiagnosePage`）、`src/components/`（`chat/`、`library/`、`ui/` 为 shadcn 组件、`AppShell`、`StatusPill`、`AuthDialog`）、`src/hooks/`、`src/lib/`（`api.ts` 请求封装，上传用 XHR 以拿到进度；`sse.ts` 极简 SSE 解析器，因为 `EventSource` 不能 POST；`auth.ts` 密钥状态；`books.ts` 书名显示美化）。

**界面功能**：

- **对话页**（`/`、`/c/:id`）：左侧会话列表（`GET /chat/sessions`，可删除）；中间流式回答（`POST /chat/stream`，可随时“停止”）；回答里的 `[n]` 变成可悬停预览的引用角标，点击在右侧“依据”面板定位原文（书名、章节、页码、片段、精排得分）；检索步骤显示为回答上方的淡色折叠行；输入框旁的“范围”可限定只在某几本书里检索。≥ 768px 会话栏常驻（可收起），< 768px 改抽屉；依据面板 ≥ 1280px 才停靠，否则是抽屉。
- **知识库页**（`/library`）：书目卡片（片段数、章节覆盖率、删除）、拖拽或点击上传（`.pdf` `.docx` `.md` `.markdown` `.txt`）、“重新扫描”，以及入库任务的实时进度（轮询 `GET /ingest/jobs/{id}`，刷新页面后仍会继续显示）。
- **状态指示**（右上角）：绿色“就绪”（索引与 Agent 都可用）、琥珀色“仅检索”（大模型未配置）、红色“未就绪 / 离线”；点开看片段数、BM25 构建时间、设备、召回模式、大模型和 Agent 状态，以及“更换密钥”。每 15 秒轮询 `/health`。
- **鉴权**：服务要求密钥时弹出“需要访问密钥”对话框；密钥通过 `/health` 校验后保存在浏览器 localStorage（`ragbase.apiKey`），之后每个请求（含流式对话和上传）自动带上；任何请求返回 401（如密钥被更换）都会清掉本地密钥并重新弹出对话框，输入框里未发送的文字不受影响。
- 主题：浅色 / 深色 / 跟随系统，保存在浏览器 localStorage。书名显示去掉了扩展名和 z-library 之类的下载站后缀（`web/src/lib/books.ts`），完整文件名在悬停提示里。
- 重新打开历史会话时，引用角标和“依据”面板可以恢复（`GET /chat/sessions/{id}` 的助手消息带 `citations`）；会话保存在内置数据库里，重启后仍在，只有手动删除才会消失（见[架构：对话 Agent](architecture.md#对话-agent)）。
- **体检页**（`/diagnose`）：在 Web 上发起体检（后台运行并显示进度）、查看最新或历史报告、删除报告、选两份报告对比，以及题集的增删改和导入 / 导出。功能说明见 [diagnostics.md](diagnostics.md#在-web-上体检)，接口见 [api.md](api.md#体检)。
- 局限：上传一次一个文件（后端同一时刻只有一个入库任务）；没有“搜索实验”页（`/search` 接口仍可直接调用）；界面文案里还有“教材”字样（面向作者自己的语料）。

## 项目结构

源码模块地图见 [architecture.md](architecture.md#源码模块地图)。仓库顶层：

```text
src/ragbase/        Python 包（src 布局）
web/                前端工程
tests/              单元测试
scripts/            start.sh · build_web.sh · calibrate_rerank_threshold.py · gen_config_docs.py
docs/               文档
Dockerfile · docker-compose.yml · .dockerignore
.env.example        配置模板（与 src/ragbase/resources/env.example 内容相同）
eval_questions.json 作者语料的评测题集（首次打开数据库时自动导入；校准脚本直接读这个文件）
bm25_userdict.txt   领域词典（作者语料；新项目由 ragbase init 生成默认的）
```

## 测试

```bash
uv run python tests/run_all.py            # 依次运行全部 tests/test_*.py，每个脚本一个子进程，打印各自汇总行；任一失败则非零退出
uv run python tests/run_all.py llm api    # 只跑文件名包含任一片段的脚本
uv run python tests/test_hybrid.py        # 单独运行一个
```

约定：

- **不用 pytest**。每个 `tests/test_*.py` 是一个**纯断言脚本**：`test_` 开头的函数用 `assert` 校验，末尾的 `__main__` 块依次调用它们、打印 `通过：<函数名>`，最后必须打印 `共 N 项测试通过`——`run_all.py` 靠这行汇总判断成功。新写测试脚本照抄现有文件的模板。
- **不加载真实模型、不调用真实大模型、不碰真实的 `data/`、索引和 `ragbase.db`**，也不访问外网：用假 embedding（确定性假向量）、假流水线、脚本化的假工具调用模型；需要 HTTP 对端时在本机 `127.0.0.1` 起**桩服务**（`test_llm.py` 的 OpenAI 兼容桩、`test_providers.py` 的 embedding / rerank 桩）；用临时目录和临时 Chroma / 本地 Qdrant；涉及数据库的测试用临时目录里的 SQLite 文件（让 `RAG_HOME` / `RAG_DB_PATH` 指向临时目录，或直接 `Database(":memory:")`）；API 用 FastAPI `TestClient` + 注入的假流水线（`create_app(pipeline_factory=…, settings=…, ingest_fn=…, chat_factory=…)`）。
- 测试里要与外部环境无关：先清掉 `RAG_*` / `ARK_*` 环境变量（参考 `test_config.py`、`test_cli_setup.py` 里的 `env()` 辅助），用 `Settings(_env_file=None)` 避免读到本机 `.env`。
- **改动哪块就跑哪块**；合并前跑全量。

| 测试脚本 | 覆盖 |
| --- | --- |
| `test_api.py` | HTTP 接口（假流水线 / 假入库）、统一错误格式、上传校验 |
| `test_auth.py` | API Key 鉴权、公开白名单、CORS、日志不泄密 |
| `test_agent.py` | 对话 Agent：工具、会话级引用、限制、回滚、历史窗口 |
| `test_db.py` | 内置 SQLite：建库、迁移、事务、并发写、错误信息、`RAG_DB_PATH`、doctor 检查 |
| `test_chat_persistence.py` | 会话持久化：重启后会话 / 历史 / 引用编号保持、历史窗口、半截轮次恢复、checkpoint 瘦身、删除、Agent 未就绪时的会话接口 |
| `test_questions.py` | 题集（`QuestionStore`）与体检报告历史（`ReportStore`）、`ragbase questions` 子命令 |
| `test_diagnose.py` | 体检核心：分阶段计时、指标、建议规则、报告 |
| `test_api_diagnose.py` | 题集 / 体检 / 报告 HTTP 接口：校验、互斥、快照、鉴权 |
| `test_web.py` | 会话列表、前端静态托管与 SPA 回退 |
| `test_llm.py` | 大模型工厂、超时 / 重试、`/ask` 端到端（本机桩服务） |
| `test_providers.py` | 云端 embedding / rerank、提供方工厂、索引签名 |
| `test_cli_setup.py` | init / models download / doctor / `.env.example` 覆盖所有配置项 |
| `test_config.py` | Settings 默认值、环境变量、别名 |
| `test_hybrid.py` · `test_sources.py` | RRF / 分词；按书过滤、快照刷新、分词缓存 |
| `test_ingest_service.py` · `test_parser.py` · `test_toc.py` · `test_fixes.py` | 增量入库；解析；PDF 书签；入库计划、评测裁判解析 |
| `test_refusal.py` · `test_context.py` | 拒答；上下文过滤与“先摘录再回答” |
| `test_vectorstores.py` | Chroma / Qdrant 同一套契约测试、迁移、工厂 |
| `test_docs.py` | 文档：配置项表是否最新、每个 Settings 字段 / 环境变量都出现在 `docs/configuration.md`、相对链接可达 |

### 文档与配置的同步

`docs/configuration.md` 里的配置项表由 `scripts/gen_config_docs.py` 从 `Settings` 生成：

```bash
uv run python scripts/gen_config_docs.py           # 更新文档里 BEGIN/END 标记之间的内容
uv run python scripts/gen_config_docs.py --check   # 只检查是否最新
```

**新增或修改 `Settings` 字段时**：在脚本的 `GROUPS`（分组）和 `DESCRIPTIONS`（说明）里补上，必要时补 `DEFAULT_TEXT`，运行脚本更新文档，并把变量加进 `src/ragbase/resources/env.example`（同步到根目录 `.env.example`）。漏做任何一步，`test_docs.py` 或 `test_cli_setup.py` 会失败。

### CI

`.github/workflows/tests.yml`，push / PR 到 `main` 时触发：

- `tests`（ubuntu）：`uv sync --locked` + `uv pip check`，然后 `uv run --locked python tests/run_all.py`。设置 `RAG_DEVICE=cpu`、`HF_HUB_OFFLINE=1`（测试不需要也不允许联网下载模型）。
- `web`（ubuntu）：`pnpm install --frozen-lockfile` → `pnpm build`（类型检查 + 打包）→ `pnpm lint`；不跑浏览器测试。

## 编码约定

- **注释、文档字符串、面向用户的消息一律用中文**；技术术语保留英文。错误信息要能直接展示给用户，且**绝不含密钥**。
- **数据库表结构只通过迁移改**：`ragbase/db/migrations.py` 的 `MIGRATIONS` 只增不改，已发布的迁移不要动，要加表 / 加列就追加一个新的迁移函数（只用 `conn.execute`，不用 `executescript`）；写库一律走 `Database.transaction()`，读用 `Database.query()`；LangGraph checkpointer 的表由它自己管理，用 `new_connection()` 的独立连接。
- **配置只有一个来源**：`ragbase/config.py` 的 `Settings`。不要在别处直接读 `os.environ`（`RAG_HOME` 除外，因为它决定 `.env` 的位置）。默认值也只在这里定义。
- **密钥用 `SecretStr`**，只在真正需要时 `get_secret_value()`；不写进日志、`repr`、错误响应、`/health`。
- **工厂是唯一的构造点**：`models.factory.create_embeddings / create_reranker`、`llm.factory.create_chat_model`、`vectorstores.create_vector_store`——调用方不要自己 new 具体实现。
- **重型依赖懒加载**：`torch`、`sentence_transformers`、`chromadb`、`qdrant_client`、`ragas` 只在用到它们的函数 / 模块里导入，所以 `ragbase --help`、`doctor`、全云端配置启动都很快。`config.py` 不能 import torch。
- **鉴权是“默认拒绝 + 公开白名单”**：新增路由默认受保护；要公开必须改 `api/auth.py` 的白名单，并补测试。
- **API 错误统一用 `{"error", "detail"}`**（抛 `HTTPException` 即可）；新增状态码要加进 `ERROR_CODES`。
- 新增路由要同步：`schemas.py` 里的请求 / 响应模型、`docs/api.md`、`web/vite.config.ts` 的代理前缀（如果前端要用）。
- 本地模型的锁：召回 + 精排必须在 `pipeline.lock` 里；新增的后台任务写向量库时同样按批持锁。
- 改了解析 / 清洗 / 切分 / 章节规则，要递增 `INGEST_PIPELINE_VERSION`（`ingest/common.py`），让老索引下次增量入库时重建；改了分词规则递增 `TOKENIZER_VERSION`（`retrieval/hybrid.py`）。

## 如何新增向量库 / 模型提供方 / LLM 协议

### 新增向量库后端

1. 在 `src/ragbase/vectorstores/<名字>.py` 实现 `VectorStore` 协议（`exists / count / upsert / delete / ids_where / all_ids / query / iter_all / reset / close` 与 `name`、`location` 属性，约定见 `vectorstores/base.py`）。**这个模块应是整个项目里唯一 import 该客户端库的地方**。`query()` 必须按“最相似在前”返回；元数据只接受标量（用 `clean_metadata`）；要能被入库线程和请求线程同时使用。
2. `config.py`：把名字加进 `VECTOR_STORES`，**同时**改 `Settings.vector_store` 的 `Literal[...]`；需要的位置 / 连接配置加成新的 Settings 字段。
3. `vectorstores/factory.py` 的 `create_vector_store` 加一个分支。`migrate-store` 的 `--to/--from` 选项取自 `VECTOR_STORES`，自动可用。
4. 测试：把新后端加进 `tests/test_vectorstores.py` 的契约测试参数里（同一套用例跑所有后端）。
5. 文档与模板：补 Settings 说明（`gen_config_docs.py`）、`env.example`、[architecture.md](architecture.md#向量库抽象) 的对比表；必要时在 `doctor.py` 的 `check_knowledge` 里加针对性的修复提示。

### 新增 embedding / reranker 提供方

1. `config.py`：在 `EMBEDDING_PROVIDERS` / `RERANKER_PROVIDERS` 和 `Settings.embedding_provider` / `reranker_provider` 的 `Literal[...]` 里加上名字。
2. 实现类：embedding 需有 `embed_documents / embed_query / embed_queries`（和 doctor 用的 `probe()`）；reranker 需有 `rerank / rerank_with_scores`（返回 `[(原始索引, 分数)]`，分数从高到低）。云端实现可复用 `models/api_client.py` 的 `ApiClient`（超时、指数退避、不泄露密钥的错误信息）。
3. `models/factory.py`：`create_embeddings` / `create_reranker` 加分支；`provider_problems` 加静态校验（地址缺失等，不联网）；如果也有云端探测，在 `doctor.py` 的 `_probe_providers` 里加。
4. 索引签名会自动包含新的提供方名；换到新提供方就需要 `ingest --full`，无需额外代码。
5. 测试放在 `tests/test_providers.py`（本机桩服务）；文档同上。

### 新增 LLM 协议

`Settings.llm_protocol` 的 `Literal` 加字面量 → `llm/factory.py` 的 `SUPPORTED_PROTOCOLS` 加名字，并在 `create_chat_model` 与 `create_ragas_llm` 各加一个分支。调用方不用改；协议无法识别时在构造阶段（服务启动）就抛 `ValueError`。对话 Agent 要求模型支持工具调用。

### 新增 Agent 工具 / 换领域文案

工具在 `agent/tools.py` 的 `build_tools` 里定义，检索类工具要复用 `retrieve_and_rerank` + `should_refuse` + `select_context_docs`，并把片段登记进会话级 `CitationRegistry`，才能保证引用编号与拒答行为和 `/ask` 一致。`agent/service.py` 的 `SYSTEM_PROMPT` 与 `agent/tools.py` 的 `SEARCH_DESCRIPTION` 目前写死了作者语料（《系统架构设计师》《信息系统项目管理师》教材）的领域文案，换领域请一并修改。
