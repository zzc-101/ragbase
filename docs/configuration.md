# 配置参考

ragbase 的全部配置集中在 [`src/ragbase/config.py`](../src/ragbase/config.py) 的 `Settings` 里，命令行、HTTP 服务和脚本共用，默认值只有这一个来源。

## 怎么配置

- **环境变量或 `.env`**：`ragbase init` 会按内置模板在项目主目录生成 `.env`（也可以 `cp .env.example .env`）。模板里每一项都分好组、带中文注释，**示例全部是注释掉的**，去掉行首 `#` 再改值即可；不写的项一律用默认值。
- **前缀**：自有配置统一用 `RAG_` 前缀（如 `RAG_PORT`）。大模型用 `RAG_LLM_*`，旧的 `ARK_*` 名字仍兼容（见下表）。
- **空值**：模型名 / 地址 / 密钥类变量，以及 `RAG_EMBEDDING_DIMENSIONS`、`RAG_EMBEDDING_TIMEOUT`、`RAG_EMBEDDING_MAX_RETRIES`、`RAG_RERANKER_TIMEOUT`、`RAG_RERANKER_MAX_RETRIES`、`RAG_INDEX_INFO_FILE`、`RAG_LLM_MAX_TOKENS`、`RAG_QDRANT_URL`、`RAG_CORS_ORIGINS` 写成空串（如 `RAG_LLM_MODEL=`）等同于没设置；**其余数字 / 枚举变量写空串会报错**（如 `RAG_PORT=`），不用时请整行注释掉。提供方、向量库后端、日志级别会去空白并忽略大小写。
- **非法取值**：枚举值不在范围内、数字超出范围（如端口、`≥ 1` 的整数）时，进程启动即抛出 pydantic 的 `ValidationError`（目前是一段 traceback，末尾指出是哪个字段）。
- **密钥安全**：所有密钥类变量（`*_API_KEY`）以 `SecretStr` 保存，日志、`repr`、`/health`、`ragbase doctor` 都不会输出明文；`doctor` 只报告“是否已设置”。
- **`.env` 与 shell**：见下面的“优先级”。运行 `ragbase doctor` 可以一屏检查当前配置是否完整。

## 优先级

1. 同一个变量名同时出现在 shell / Docker 环境和 `.env` 里：**环境变量优先**，`.env` 里的值不生效。
2. 命令行入口（`ragbase …`）启动时会先把 `.env` 载入进程环境（`python-dotenv`，**不覆盖已存在的变量**），再读取 Settings。`ragbase.ingest.common` 在导入时也会做同样的事，所以 `uvicorn ragbase.api.app:app` 这种启动方式同样读 `.env`。
3. 大模型的三项（`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`）有新旧两套名字，**新名 `RAG_LLM_*` 总是优先于旧名 `ARK_*`，不管它们各自来自 shell 还是 `.env`**（实测：shell 里 `ARK_CHAT_MODEL=a`、`.env` 里 `RAG_LLM_MODEL=b`，通过命令行 / 服务启动时生效的是 `b`）。例外是不经过 `load_env()` 直接构造 `Settings()` 的程序（如自己写的脚本）：此时 shell 的 `ARK_*` 会压过 `.env` 里的 `RAG_LLM_*`。迁移时请把旧变量改名，不要两套混用。
4. 命令行参数（`ragbase ask --min-score` 等）和 API 请求字段优先于 Settings 默认值。

## 配置项一览

下表按用途分组，涵盖 `Settings` 的**全部**字段。

<!-- BEGIN GENERATED:settings -->

> 本节由 `scripts/gen_config_docs.py` 从 `ragbase.config.Settings` 生成，请勿手改；改了 Settings 后运行 `uv run python scripts/gen_config_docs.py` 更新。

### 路径与项目主目录

所有运行时数据默认都放在**项目主目录**下：环境变量 `RAG_HOME`，未设置则为**当前工作目录**。`.env`、`bm25_userdict.txt`、`eval_questions.json`、`.ragas_cache/`、`eval_results.*.json` 的位置也由它决定（它们不是 Settings 字段，没有单独的变量）。

`RAG_HOME` 必须是真正的环境变量（shell / Docker 的 `ENV`），**不能写在 `.env` 里**——`.env` 自己的位置就依赖它。下表的路径变量建议写绝对路径：除 `RAG_WEB_DIST` 外，相对路径不会按 `RAG_HOME` 解析（而是相对进程当前目录）。

| 环境变量 | 类型 / 取值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_DATA_DIR` | 路径 | `<RAG_HOME>/data` | 文档目录。`POST /documents` 上传到这里，`ragbase ingest` / `POST /ingest` 扫描这里（只扫顶层，不递归）。 |
| `RAG_DB_DIR` | 路径 | `<RAG_HOME>/chroma_db` | Chroma 持久化目录（仅 `RAG_VECTOR_STORE=chroma` 时使用）。 |
| `RAG_META_FILE` | 路径 | `<RAG_HOME>/.ingest_meta.json` | 增量入库的文件指纹记录。 |
| `RAG_INDEX_INFO_FILE` | 路径 | （空，即 `.index_info.json`，与 `RAG_META_FILE` 同目录） | 索引的 embedding 签名文件（提供方 / 模型 / 维度）。留空 = 与 `RAG_META_FILE` 同目录的 `.index_info.json`。 |
| `RAG_CACHE_DIR` | 路径 | `<RAG_HOME>/.bm25_cache` | BM25 分词缓存目录。 |
| `RAG_DB_PATH` | 路径 | （空，即 `<RAG_HOME>/ragbase.db`） | 内置 SQLite 数据库文件（对话会话、评测题集、体检报告都存在里面；WAL 模式，旁边会出现 `-wal` / `-shm` 文件）。留空 = `<RAG_HOME>/ragbase.db`，父目录不存在时自动创建；目录不可写时 `ragbase doctor` 报 ✗。 |

### 检索与拒答

这些值是服务和命令行的默认值；`/search`、`/ask` 的请求字段与 `ragbase ask` / `ragbase eval` 的命令行参数可按次覆盖。

| 环境变量 | 类型 / 取值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_MODE` | `vector` / `bm25` / `hybrid` | `hybrid` | 默认召回方式：`vector` 仅向量、`bm25` 仅关键词、`hybrid` 两路 RRF 融合。 |
| `RAG_RETRIEVE_K` | 整数，≥ 1 | `20` | 精排前的候选数（混合检索是 RRF 融合后保留的条数；单路模式是该路取的条数）。 |
| `RAG_PER_ROUTE_K` | 整数，≥ 1 | `30` | 混合检索时向量路和 BM25 路各取多少条参与融合。 |
| `RAG_RERANK_K` | 整数，≥ 1 | `5` | 精排后保留的片段数（`/search` 的 `top_k` 默认值，也是 `/ask`、对话 Agent 每次检索保留的条数）。 |
| `RAG_MIN_SCORE` | 数字，≥ 0，≤ 1 | `0.7` | 拒答阈值：最相关片段的精排得分低于它就拒答、**不调用大模型**；`0` 表示不拒答。默认值按默认本地 reranker 标定，换 reranker 要重新标定；`RAG_RERANKER_PROVIDER=none` 时没有得分，不生效。 |
| `RAG_CONTEXT_MIN_SCORE` | 数字，≥ 0，≤ 1 | `0.95` | 只把精排得分不低于它的片段交给大模型（得分最高的 top1 总是保留）；`0` 表示不过滤。得分已高度饱和，不要盲目调高。 |

### 向量库

后端的取舍、迁移与本地 / 服务端模式的区别见 [architecture.md](architecture.md#向量库抽象) 与 [deployment.md](deployment.md)。

| 环境变量 | 类型 / 取值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_VECTOR_STORE` | `chroma` / `qdrant` | `chroma` | 向量库后端。取值不在列表内时启动报错。 |
| `RAG_COLLECTION` | 字符串，非空 | `rag` | 集合名（Chroma collection / Qdrant collection）。 |
| `RAG_QDRANT_PATH` | 路径 | `<RAG_HOME>/qdrant_db` | Qdrant **本地嵌入式**模式的数据目录（`RAG_QDRANT_URL` 未设置时使用）。同一目录同一时刻只允许一个进程打开。 |
| `RAG_QDRANT_URL` | 字符串 | （空，用本地目录） | Qdrant **服务端**地址，如 `http://localhost:6333`（Docker Compose 里是 `http://qdrant:6333`）；设置后忽略 `RAG_QDRANT_PATH`。 |
| `RAG_QDRANT_API_KEY` | 密钥 | （空） | Qdrant 服务端的 API Key（可选；不会出现在日志 / `repr` 里）。 |

### embedding

提供方、示例组合与换模型的规则见 [models.md](models.md)。

| 环境变量 | 类型 / 取值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_EMBEDDING_PROVIDER` | `local` / `openai` | `local` | `local`：sentence-transformers 加载本地模型；`openai`：OpenAI 兼容的 `POST {地址}/embeddings`。换提供方 / 模型 / 维度必须 `ragbase ingest --full` 重建索引。 |
| `RAG_EMBEDDING_MODEL` | 字符串，非空 | `Qwen/Qwen3-Embedding-0.6B` | 本地：Hugging Face 仓库名；云端：服务商的模型 ID。 |
| `RAG_EMBEDDING_BASE_URL` | 字符串 | （空，回退 `RAG_LLM_BASE_URL`） | 云端 embedding 的接口根地址（会请求 `{地址}/embeddings`，须以 `http://` 或 `https://` 开头）。留空回退到 `RAG_LLM_BASE_URL`。 |
| `RAG_EMBEDDING_API_KEY` | 密钥 | （空，同源时回退 `RAG_LLM_API_KEY`） | 云端 embedding 的密钥。留空时，仅当最终地址与 `RAG_LLM_BASE_URL` 是同一个服务（协议 + 主机 + 端口相同）才沿用 `RAG_LLM_API_KEY`，避免把 A 服务商的密钥发给 B 服务；本机 Ollama / vLLM 等不校验密钥的服务可以不填。 |
| `RAG_EMBEDDING_DIMENSIONS` | 整数，≥ 1 | （空，不发送 `dimensions`） | 可选：作为 `dimensions` 参数发给服务端（仅支持缩减维度的模型）；也计入索引签名。服务端返回的向量长度与它不符时入库会报错。本地模型忽略它。 |
| `RAG_EMBEDDING_BATCH_SIZE` | 整数，≥ 1 | `16` | 云端 embedding 每次请求最多发多少条文本；部分服务商有上限，请以服务商文档为准。 |
| `RAG_EMBEDDING_QUERY_INSTRUCTION` | 字符串 | （空） | 可选：只加在“查询”前面的指令前缀，文档不加（如 Qwen3-Embedding 的 `Instruct: ...\nQuery: `，其中两个字符 `\n` 会还原成换行）。本地模型自带 query prompt，不用它。 |
| `RAG_EMBEDDING_TIMEOUT` | 数字，> 0 | （空，沿用 `RAG_LLM_TIMEOUT`） | 云端 embedding 单次请求超时（秒）。留空沿用 `RAG_LLM_TIMEOUT`。 |
| `RAG_EMBEDDING_MAX_RETRIES` | 整数，≥ 0 | （空，沿用 `RAG_LLM_MAX_RETRIES`） | 云端 embedding 超时 / 连接错误 / 429 / 5xx 的重试次数（指数退避）。留空沿用 `RAG_LLM_MAX_RETRIES`。 |

### reranker

提供方与换 reranker 后需要重新标定阈值的规则见 [models.md](models.md)。

| 环境变量 | 类型 / 取值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_RERANKER_PROVIDER` | `local` / `api` / `none` | `local` | `local`：CrossEncoder 加载本地模型；`api`：通用 `POST {地址}/rerank`；`none`：不精排（保持召回顺序，没有得分，**不会按相关度拒答**，只有召回为空才拒答）。 |
| `RAG_RERANKER_MODEL` | 字符串，非空 | `Qwen/Qwen3-Reranker-0.6B` | 本地：Hugging Face 仓库名；云端：服务商的模型 ID。`none` 时不使用。 |
| `RAG_RERANKER_BASE_URL` | 字符串 | （空，回退 `RAG_LLM_BASE_URL`） | 云端 rerank 的接口根地址（会请求 `{地址}/rerank`）。留空回退到 `RAG_LLM_BASE_URL`。 |
| `RAG_RERANKER_API_KEY` | 密钥 | （空，同源时回退 `RAG_LLM_API_KEY`） | 云端 rerank 的密钥。留空时的回退规则同 `RAG_EMBEDDING_API_KEY`。 |
| `RAG_RERANKER_TIMEOUT` | 数字，> 0 | （空，沿用 `RAG_LLM_TIMEOUT`） | 云端 rerank 单次请求超时（秒）。留空沿用 `RAG_LLM_TIMEOUT`。 |
| `RAG_RERANKER_MAX_RETRIES` | 整数，≥ 0 | （空，沿用 `RAG_LLM_MAX_RETRIES`） | 云端 rerank 的重试次数。留空沿用 `RAG_LLM_MAX_RETRIES`。 |

### 本地模型设备

只对提供方为 `local` 的 embedding / reranker 生效；全云端时 `/health` 的 `device` 显示 `cloud`。

| 环境变量 | 类型 / 取值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_DEVICE` | `auto` / `mps` / `cuda` / `cpu` | `auto` | 本地模型设备：`auto` 依次尝试 mps（Apple Silicon）、cuda、cpu；mps / cuda 上用 fp16，cpu 保持默认精度。 |

### 大模型（回答与对话）

走 OpenAI 兼容协议。`RAG_LLM_*` 是正式名，旧的 `ARK_*` 名字（火山方舟时代遗留）仍然兼容。三项（地址 / 密钥 / 模型名）缺任何一项都算“未配置”：服务仍可启动、`/search` 可用，`/ask` 与 `/chat` 返回 503。

| 环境变量 | 类型 / 取值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_LLM_PROTOCOL` | `openai` | `openai` | 大模型协议。目前只实现 `openai`（OpenAI 兼容的 `/chat/completions`）；新增协议见 [development.md](development.md#如何新增向量库--模型提供方--llm-协议)。 |
| `RAG_LLM_BASE_URL`<br>（旧名 `ARK_BASE_URL`） | 字符串 | （空） | 接口根地址，如 `https://api.openai.com/v1`。 |
| `RAG_LLM_API_KEY`<br>（旧名 `ARK_CHAT_API_KEY`） | 密钥 | （空） | API Key（以 `SecretStr` 保存，不会出现在日志 / `repr` 里）。 |
| `RAG_LLM_MODEL`<br>（旧名 `ARK_CHAT_MODEL`） | 字符串 | （空） | 模型名。 |
| `RAG_LLM_TEMPERATURE` | 数字，≥ 0，≤ 2 | `0` | 采样温度。 |
| `RAG_LLM_MAX_TOKENS` | 整数，≥ 1 | （不设置，不限制） | 单次回答的最大输出 token 数；不设置或留空 = 不限制，用服务端默认。 |
| `RAG_LLM_TIMEOUT` | 数字，> 0 | `60` | 单次请求超时（秒）。 |
| `RAG_LLM_MAX_RETRIES` | 整数，≥ 0 | `2` | 超时 / 连接错误 / 429 / 5xx 的自动重试次数（指数退避），最坏等待约 `TIMEOUT × (RETRIES + 1)` 加退避间隔。 |

### 评测裁判模型（可选）

`ragbase eval` 的 Ragas 与中文裁判可以用另一个（通常更强的）模型；设置了哪项就覆盖哪项，其余沿用上面的回答模型。

| 环境变量 | 类型 / 取值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_EVAL_LLM_BASE_URL` | 字符串 | （空，沿用回答模型） | 评测裁判的接口地址；留空沿用回答模型的。 |
| `RAG_EVAL_LLM_API_KEY` | 密钥 | （空，沿用回答模型） | 评测裁判的密钥；留空沿用回答模型的。 |
| `RAG_EVAL_LLM_MODEL` | 字符串 | （空，沿用回答模型） | 评测裁判的模型名；留空沿用回答模型的。 |

### 对话 Agent

会话（消息、引用、会话列表）保存在内置数据库（见上面的 `RAG_DB_PATH`）里，重启不丢，也不会自动淘汰，只有在 Web 或 `DELETE /chat/sessions/{id}` 里删除才会清掉。机制见 [architecture.md](architecture.md#对话-agent)。

| 环境变量 | 类型 / 取值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_AGENT_MAX_TOOL_CALLS` | 整数，≥ 1 | `4` | 每轮用户消息最多调用几次工具；超出后工具被拒绝，模型只能据已有结果作答。 |
| `RAG_AGENT_MAX_MODEL_CALLS` | 整数，≥ 2 | `6` | 每轮用户消息最多调用几次大模型；超出后直接结束并返回提示“本轮处理步骤过多”。 |
| `RAG_CHAT_HISTORY_TURNS` | 整数，≥ 1 | `10` | 发给大模型的最近几轮对话（一轮 = 一条用户消息及其全部回复 / 工具消息；整轮截取，不额外调用大模型做摘要）。只影响大模型看到的上下文，库里和 Web 上仍保留会话的全部消息。 |

### 服务、鉴权与 CORS

鉴权与暴露方式的完整说明见 [deployment.md](deployment.md#安全清单) 与 [api.md](api.md#鉴权)。

| 环境变量 | 类型 / 取值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_HOST` | 字符串 | `127.0.0.1` | `ragbase serve` 的监听地址。`0.0.0.0` = 对外开放（务必同时设置 `RAG_API_KEY`）。 |
| `RAG_PORT` | 整数，≥ 1，≤ 65535 | `8000` | `ragbase serve` 的监听端口。 |
| `RAG_API_KEY` | 密钥 | （空，不鉴权） | 接口共享密钥：设置后所有 API 请求都要带它（`Authorization: Bearer <key>` 或 `X-API-Key`）；留空 = 不鉴权。 |
| `RAG_CORS_ORIGINS` | 逗号分隔列表 | （空，不加 CORS） | 允许跨域访问的前端来源，逗号分隔（如 `https://app.example.com,http://localhost:5173`）；留空 = 不加 CORS 中间件。 |
| `RAG_MAX_UPLOAD_MB` | 整数，≥ 1 | `300` | `POST /documents` 的单文件大小上限（MB）；超限返回 413。 |
| `RAG_LOG_LEVEL` | 字符串 | `INFO` | 日志级别：`DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`（不区分大小写）。 |

### Web 界面

| 环境变量 | 类型 / 取值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_WEB_DIST` | 路径 | `<RAG_HOME>/web/dist`（没有则用随包副本） | 前端构建产物目录。默认 `<RAG_HOME>/web/dist`，没有则用随包分发的副本；相对路径按 `RAG_HOME` 解析。目录存在时由 API 服务在同一端口托管。 |

### 已废弃的配置项

下面的配置项仍会被解析（旧 `.env` 不会因此启动报错），但**不再生效**；设置了它们，服务启动日志里会给一条“已废弃”的警告。请从 `.env` 中删除。

| 环境变量 | 类型 / 取值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `RAG_CHAT_MAX_SESSIONS`<br>（已废弃） | 整数，≥ 1 | `200` | **已废弃，不再生效。** 原来是内存中最多保留的会话数；会话持久化后不再按数量淘汰。 |
| `RAG_CHAT_SESSION_TTL`<br>（已废弃） | 数字，> 0 | `3600` | **已废弃，不再生效。** 原来是会话空闲多少秒后过期；会话持久化后不再按空闲时间过期。 |

<!-- END GENERATED:settings -->

## 云端模型接口的回退规则

云端 embedding / rerank（`RAG_EMBEDDING_PROVIDER=openai` / `RAG_RERANKER_PROVIDER=api`）的连接参数这样确定：

- **地址**：`RAG_EMBEDDING_BASE_URL` / `RAG_RERANKER_BASE_URL` 留空时用 `RAG_LLM_BASE_URL`（很多服务商同一个地址同时提供大模型、embedding、rerank）。
- **密钥**：自己的密钥留空时，**只有最终地址与 `RAG_LLM_BASE_URL` 是同一个服务（协议 + 主机 + 端口相同）才沿用 `RAG_LLM_API_KEY`**；单独指定了别的地址就必须单独填密钥，否则请求不带 `Authorization` 头（本机 Ollama / vLLM 不校验密钥的可以不填）。`ragbase doctor` 会提示“未设置密钥”。
- **超时与重试**：留空沿用 `RAG_LLM_TIMEOUT` / `RAG_LLM_MAX_RETRIES`。重试只针对超时、连接错误、HTTP 408 / 409 / 425 / 429 / 500 / 502 / 503 / 504，间隔指数退避（0.5s、1s、2s…）。报错信息里不含密钥。
- 启动前的静态检查（地址缺失、不是 `http(s)://` 开头）由 `provider_problems()` 完成，不联网；`ragbase serve / ingest / ask / chat` 发现问题会直接退出并给出中文说明。

## 其他环境变量（不是 Settings 字段）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAG_HOME` | 当前工作目录 | 项目主目录，见上文“路径与项目主目录”。必须是真环境变量，不能写在 `.env` 里。Docker 镜像里是 `/data`。 |
| `HF_ENDPOINT` | `https://huggingface.co` | Hugging Face 端点；国内网络用 `https://hf-mirror.com`。`ragbase models download`、`serve --download-models` 以及本地模型加载都遵循它。写进 `.env` 也有效（命令行入口会先载入 `.env`）。Docker Compose 里它被显式传给容器。 |
| `HF_HOME` | `~/.cache/huggingface` | 本地模型缓存根目录。Docker 镜像里是 `/models`（挂载 `ragbase-models` 卷）。 |
| `CHROMA_TELEMETRY_DISABLED` | 模板里为 `true` | 关闭 Chroma 的遥测。`.env` 模板和 Docker 镜像都设置了它。 |
| `RAGBASE_BIND` / `RAGBASE_PORT` | `127.0.0.1` / `8000` | **仅 `docker-compose.yml` 使用**：宿主机上发布的绑定地址和端口（容器内始终监听 `0.0.0.0:8000`）。 |
| `NO_COLOR` | 未设置 | 设置后 `ragbase doctor` 与 `scripts/start.sh` 不输出彩色。 |

## 另见

- [models.md](models.md)：本地 / 云端模型组合、换模型的规则、阈值校准
- [deployment.md](deployment.md)：Docker、反向代理、安全清单
- [troubleshooting.md](troubleshooting.md)：配置相关的常见问题
