# 常见问题与排错

先跑一遍体检，大多数配置问题它会直接指出来：

```bash
uv run ragbase doctor            # Docker：docker compose run --rm ragbase doctor
```

## 先用 `ragbase doctor`

一屏检查：Python 版本、项目主目录与文档目录是否可写（并显示 `RAG_HOME` 是否设置）、`.env`、内置数据库（路径、能否打开、表结构版本是否最新）、embedding / reranker 的提供方与模型（本地：计算设备与模型缓存；云端：地址 / 密钥是否配置）、向量库是否可访问及片段数、索引的 embedding 签名是否与配置一致、是否需要 `ragbase ingest`、大模型是否配置（只显示协议 / 模型 / 地址，不显示密钥）、接口鉴权、Web 资源。每项给出 `✓`（正常）/ `!`（警告）/ `✗`（必须修复）和修复提示；**有 `✗` 时退出码非零**。

- 默认**不调用**大模型，也不调用云端 embedding / rerank。
- `--check-llm`：额外发一次最小的大模型请求，验证地址 / 密钥 / 模型名。
- `--check-models`：对每个云端 embedding / rerank 接口各发一次最小请求。
- 两者都会产生极少量 API 调用；报错里的密钥会被抹成 `***`。

## 模型接口返回 4xx / 模型端点已关闭 / 模型 ID 写错

**现象**：`/ask`、`/chat` 返回 502（`llm_error`，响应里只有固定提示，真正的异常在服务日志里）；`ragbase ask` 抛出 `GenerationError`（消息里带原因类名，如 `AuthenticationError`）；云端 embedding / rerank 的报错形如 “embedding 接口返回 HTTP 404（检查地址 / 模型名是否正确）：…”；`ragbase ingest` 在向量化阶段失败。

**排查**：

1. `ragbase doctor --check-llm`（大模型）/ `--check-models`（云端 embedding / rerank）直接复现并给出脱敏后的错误。
2. **模型 ID 的写法因服务商而异**，写错通常是 404 / 400。例如火山方舟的模型 ID 用连字符分隔并带日期后缀，推理接入点则是 `ep-` 开头的 ID——不要凭记忆拼写。不确定时先让服务商告诉你有哪些模型：
   ```bash
   curl -H "Authorization: Bearer $RAG_LLM_API_KEY" "$RAG_LLM_BASE_URL/models"
   ```
   （大多数 OpenAI 兼容服务提供这个接口，是否可用取决于服务商；云端 embedding / rerank 同理，把地址换成对应的 `RAG_EMBEDDING_BASE_URL` / `RAG_RERANKER_BASE_URL`。）
3. **模型 / 接入点在服务商控制台里被关闭、下线或没有开通**（或账号余额 / 额度用尽）也会表现为类似的 4xx：去控制台确认该模型处于启用状态。
4. 地址要写到 API 根路径，通常以 `/v1` 结尾（方舟是 `/api/v3`）；ragbase 会在后面拼 `/chat/completions`、`/embeddings`、`/rerank`。返回的不是 JSON 时，报错会提示“请确认地址以 /v1 之类的 API 根路径结尾”。
5. 401 / 403 看密钥；云端 embedding / rerank 地址与大模型不同源时，**不会**沿用 `RAG_LLM_API_KEY`，要单独填（[规则](configuration.md#云端模型接口的回退规则)）。

> 具体厂商的地址和模型 ID 在本文档和模板里都只是示例，没有逐一验证，请以服务商文档为准。

## 提示“当前 embedding 配置与索引不一致”

换过 embedding 的提供方 / 模型 / 维度后，旧向量不能再用，`serve` / `ask` / `chat` / 增量 `ingest`（以及 `eval`）会拒绝启动，`doctor` 报 `✗`。

- 想用新的 embedding：停服务，运行 `ragbase ingest --full` 重建索引。
- 想继续用旧索引：把 embedding 配置改回建索引时的值（错误信息里两边的值都列出来了）。
- 没有 `.index_info.json` 的旧索引视为默认本地模型建的：配置恰好是默认值就自动补记，否则拒绝。文件损坏同样拒绝；确认无误后可删除该文件。

规则细节见 [models.md](models.md#换-embedding索引签名规则)。

## 在别的目录运行，找不到索引 / 配置不生效（`RAG_HOME`）

数据、索引、`.env` 的默认位置都从**项目主目录**推导：环境变量 `RAG_HOME`，未设置则为**当前工作目录**。在别的目录运行 `ragbase` 会得到“索引不存在或为空”、`doctor` 里 “RAG_HOME 未设置，使用当前目录” 的提示、读不到 `.env`。

- 回到项目目录运行，或 `export RAG_HOME=/path/to/project`；`ragbase init [DIR]` 的“下一步”提示里也会给出这条命令。
- `RAG_HOME` 必须是**真正的环境变量**，写在 `.env` 里无效（`.env` 的位置本身依赖它）。
- Docker 里它是 `/data`（挂载自 `./ragbase-home`）。

## 嵌入式 Qdrant 报 “already accessed by another instance”

嵌入式 Qdrant（只设了 `RAG_VECTOR_STORE=qdrant`、没设 `RAG_QDRANT_URL`）同一个目录同一时刻只允许一个进程打开。`ragbase serve` 运行时再运行 `ragbase ingest` / `ask` / `migrate-store` 会报这个错。

- 用服务的在线入库接口（`POST /documents`、`POST /ingest`），或先停服务再跑命令行。
- 需要多进程访问：改用 Qdrant 服务端模式（设置 `RAG_QDRANT_URL`，见 [deployment.md](deployment.md#可选qdrant-服务端做向量库)）。
- 如果进程异常退出后仍报错，确认没有残留的 ragbase 进程；`ragbase doctor` 在这种情况下会提示“若 ragbase serve / ingest 正在运行，请先停止”。

## CPU 上精排很慢

纯 CPU 能运行，但精排明显更慢：每个问题要对约 `RAG_RETRIEVE_K`（默认 20）个候选片段逐一打分；首次建索引的大 PDF 也会很久。`doctor` 会对此给出警告。

- 改用云端 reranker / embedding（[models.md](models.md#示例组合)）——不用下载模型，全云端时也不再加载 torch。
- 或者调小 `RAG_RETRIEVE_K`（精排耗时与候选数大致成正比，代价是召回面变窄）；或设 `RAG_RERANKER_PROVIDER=none` 不精排（**没有相关度拒答**，检索质量也会下降）。
- 有 Apple Silicon（MPS）或 NVIDIA GPU（CUDA）会快很多；`RAG_DEVICE` 可指定设备，MPS 不可用时设 `RAG_DEVICE=cpu`。

## Web 界面一直要密钥 / 请求返回 401

启用了鉴权（设置了 `RAG_API_KEY`）后，所有接口都要带密钥；Web 界面会弹出“需要访问密钥”对话框。

- 密钥就是 `.env`（或环境变量）里的 `RAG_API_KEY`：`grep RAG_API_KEY .env` 可查看。`ragbase init --api-key` 新建 `.env` 时会生成一把。
- “密钥无效或已更换”：服务端的密钥改过（改完需重启服务；Docker 里用 `docker compose up -d` 重建容器才会重新读取 `env_file`）。在对话框里输入新密钥，或右上角状态弹层里点“更换密钥”。
- 用 curl 验证：`curl -H "Authorization: Bearer $RAG_API_KEY" http://127.0.0.1:8000/documents`；`/health` 带有效密钥才返回完整内容（`auth_required` 为 `true`）。
- 经过反向代理仍然 401：确认代理没有丢掉 `Authorization` / `X-API-Key` 请求头。
- 不想鉴权（仅本机使用）：清空 `RAG_API_KEY` 并重启。

## 索引为空、服务 503、旧版索引迁移

- **索引为空**：服务仍会启动，`/health` 报 `not_ready`，`/search`、`/ask`、`/chat*`、`/documents*`、`/ingest` 返回 503 “索引不存在或为空，请先运行 ragbase ingest --full”。把文档放进 `data/`，运行 `ragbase ingest`，**然后重启服务**（在线入库要求服务启动时索引已存在）。
- **文档放进去了但搜不到**：`ragbase ingest` 只扫描 `data/` 顶层；`.doc` 会被跳过（另存为 `.docx`）；扫描版 PDF 没有文本层，会提示“文档没有可索引文本，扫描版 PDF 需要 OCR”——需要先自行做 OCR（ragbase 不会自动 OCR）。
- **旧版索引**（片段没有 `fid`、ID 是 UUID，或索引非空但没有 `.ingest_meta.json`）：第一次运行增量 `ragbase ingest`（或 `POST /ingest`、上传）时会迁移——先按 ID 分批清空旧片段（不删除集合），再逐文件重建，耗时与全量入库相当。迁移前 `doctor` 会警告“索引非空但没有入库记录”。旧索引不重建也能继续使用，只是章节标题（`heading_path`）全部为 `null`。
- **入库中断 / 部分文件失败**：成功的文件已落盘记录，失败的文件下次重试；命令行以退出码 1 结束，任务状态为 `failed`。

## `.env` 与 shell 环境变量，谁生效（`RAG_LLM_*` / `ARK_*`）

- 同名变量：**shell / Docker 环境里的优先于 `.env`**。改了 `.env` 不生效时，先查 `env | grep -E '^(RAG_|ARK_|HF_)'`，很可能是 shell 里导出过同名变量；Docker 里还要看 compose 的 `environment` / `env_file`。
- 新旧大模型变量名：`RAG_LLM_*` 总是优先于旧的 `ARK_*`——通过命令行 / 服务启动时，即使 `ARK_*` 在 shell 里、`RAG_LLM_*` 在 `.env` 里，生效的也是 `RAG_LLM_*`（`.env` 先被载入进程环境，再按名字优先级取）。迁移时请把旧变量改名，**不要两套混用**。
- 检查当前生效的大模型配置：`ragbase doctor` 的 “大模型” 一组（显示协议 / 模型 / 地址）。

完整的优先级说明见 [configuration.md](configuration.md#优先级)。

## `ragbase serve` 提示“本地模型尚未下载”

运行 `ragbase models download`（国内网络加 `HF_ENDPOINT=https://hf-mirror.com`）；或让 `serve` 自动下载：`ragbase serve --download-models`。模型下载很慢 / 超时：见 [models.md](models.md#国内网络hugging-face-镜像)；也可以在能访问 Hugging Face 的机器上下载后，把整个 `HF_HOME` 目录（或 Docker 的 `ragbase-models` 卷）拷过来。已缓存的模型会跳过，下载中断后重跑可续传。embedding 与 reranker 都改用云端 API 则完全不需要下载。

## 总是拒答，或者从不拒答

- 换过 reranker（模型或云端服务商）：分数分布变了，默认的 `RAG_MIN_SCORE=0.7` 不再适用——运行 `scripts/calibrate_rerank_threshold.py` 重新标定（[models.md](models.md#reranker-换模型后重新标定拒答阈值)）。
- 看实际得分：`POST /search` 的响应里有 `top_score` 和每个片段的 `rerank_score`、`refused`；命令行 `ragbase ask` 拒答时会打印最相关片段的得分。临时试值：`ragbase ask --min-score 0.3`（`0` 表示不拒答）。
- `RAG_RERANKER_PROVIDER=none` 时没有得分，**永远不会按相关度拒答**（只有召回为空才拒答）。
- 库外问题没被拦下：本地 reranker 对少数“似是而非”的问题给分偏高，阈值拦不住，见 [models.md](models.md#默认阈值是怎么来的本地-qwen3-reranker-06b)。

## 对话流式输出卡住 / 一次性蹦出来

`/chat/stream` 是 SSE，经过的反向代理或负载均衡必须关闭响应缓冲、不压缩、超时足够长。nginx 的写法见 [deployment.md](deployment.md#反向代理与-https)。用 `curl -N` 直连服务可以确认服务端本身是否正常流式。

## 对话（Agent）不工作

- 大模型未配置：`/chat` 返回 503，`/health` 里 `agent_ready` 为 `false`，Web 界面状态为“仅检索”。
- 对话要求模型支持工具调用（`tools` / `tool_calls`）。单元测试用的是假模型，**没有用真实大模型验证过**，如果你的服务商或模型不支持函数调用，会表现为 502 或回答异常；此时可以用不依赖工具调用的 `/ask`。
- 会话不见了 / `GET /chat/sessions/{id}` 返回 404：见下一节[会话不见了](#会话不见了)。

## 数据库打不开 / 版本过新

会话、评测题集和体检报告都存在内置 SQLite 数据库里（默认 `RAG_HOME/ragbase.db`，见[架构：数据库](architecture.md#数据库)）。`ragbase serve` 启动时先打开它，打不开会直接给出中文错误并退出，不会去加载模型。

- **无法打开数据库 / 无法创建数据库目录 / 数据库不可写**：检查路径所在目录的权限和磁盘空间。WAL 模式要求**所在目录也可写**（运行时旁边会出现 `-wal` / `-shm` 文件）。Docker 里确认 `ragbase-home` 对容器内 uid 1000 可写。路径不合适时用 `RAG_DB_PATH` 指向一个可写的本地位置。
- **文件已损坏**（错误里有 `file is not a database` 之类的原文）：先把 `ragbase.db`（连同 `-wal` / `-shm`）拷走留底，再删除它，重启后会自动建一个新库——**会丢失里面的会话和体检报告**，题集会在首次启动时从 `RAG_HOME/eval_questions.json` 重新导入（文件还在的话）。有备份的话从备份恢复，见[部署：备份](deployment.md#备份)。
- **数据库版本比当前程序更新**（“数据库 … 的版本（N）比当前程序支持的（M）更新”）：这个库被更新版本的 ragbase 升级过，旧程序不能降级使用它，也不会去改它。升级 ragbase 到不低于那个版本，或者用 `RAG_DB_PATH` 换一个库。
- `ragbase doctor` 的“数据库”一组会给出同样的判断：`✗` 表示打不开 / 不可写 / 版本过新；`!` 表示表结构版本较旧，下次启动时会自动升级（升级前建议先备份）；库还不存在是正常的，首次运行 `ragbase serve` 时自动创建。

## `database is locked`

SQLite 同一时刻只允许一个写入者。ragbase 的写事务在进程内串行，并设置了 5 秒的等待（`busy_timeout`），正常使用不会碰到它；真的出现（接口返回 500，服务日志里有 `database is locked`），通常是：

- **另一个进程在长时间占用数据库**：比如命令行的 `ragbase diagnose` / `eval` / `questions` 与服务同时在写，或者一个 `sqlite3` 命令行 / 数据库浏览器开着事务没提交。关掉它们再试。
- **数据库放在不支持 WAL 的文件系统上**：NFS / SMB 等网络盘，以及 macOS 上 Docker Desktop 的绑定目录（`./ragbase-home`），文件锁和共享内存不可靠（见[部署：备份](deployment.md#备份数据库ragbasedb)里的说明）。把 `RAG_DB_PATH` 指向本机磁盘，或在 Docker 里改用命名卷。
- 同时跑了多个 `ragbase serve` 进程（服务必须单进程）。

## 会话不见了

对话会话持久化在数据库里，**只会被手动删除**，不会因为重启、空闲或数量多而消失。看不到会话时依次排查：

1. **是不是被删了**：Web 上删除会话，或 `DELETE /chat/sessions/{id}`，都是永久删除，无法恢复（除非从备份恢复 `ragbase.db`）。
2. **连的是不是同一个数据库**：库的位置是 `RAG_DB_PATH`，没设置就是 `RAG_HOME/ragbase.db`，而 `RAG_HOME` 没设置时是**当前工作目录**。换了目录启动、改了 `RAG_HOME` 或 `RAG_DB_PATH`、Docker 换了挂载目录，都会连到一个新的空库。运行 `ragbase doctor` 看“数据库”那一行显示的路径，和以前的对一下；旧库文件还在的话，把 `RAG_DB_PATH` 指回去即可。
3. **是不是命令行 `ragbase chat` 里的会话**：命令行对话只存在内存里，不写库，也不会出现在 Web 的会话列表里。
4. **第一轮就失败了**：会话在第一轮**成功**后才会出现；第一轮报错（大模型超时等）不会留下会话。
5. **升级前的内存会话**：以前版本把会话放在内存里，重启就丢，升级后找不回来。
6. **库被重建过**：数据库损坏后删除重建（见上），里面的会话随之丢失。

另外，大模型未配置或索引为空时，会话列表、历史和删除仍然可用（直接读库），只有发消息返回 503；如果这时列表是空的，多半是上面第 2 条。

## 体检与题集

- **开始体检返回 400**：题集是空的——在 Web 体检页添加题目，或 `ragbase questions import FILE`。题集在数据库里；**重启后题目没有重新出现**是正常的：`eval_questions.json` 只在数据库首次导入时读取一次，之后（包括你把题目删光之后）不会再导入。需要的话手动导入。
- **改了 `RAG_HOME/eval_questions.json` 但没有生效**：同上，文件只是导入 / 导出的格式，已经导入过的数据库不会再读它；用 `ragbase questions import FILE --replace` 替换。
- **开始体检返回 409**：已有体检、入库任务或删除文档在运行，等它结束再试；**返回 503**：索引为空或服务还没加载好流水线。
- **导入题集返回 422 / 命令行报“题集校验失败”**：整个文件都没有导入，`errors`（命令行是逐条打印）列出了第几题（命令行显示的序号从 1 起，接口 `errors` 里的 `index` 从 0 起）、哪个字段有问题；改好后重新导入。
- **体检期间对话变慢**：预期行为，体检复用服务里的模型，每题单独占一次检索锁。报告里“体检期间有对话请求”的耗时数字偏高，对比时会有提示。
- **`ragbase eval` 说有题被跳过**：这些题的 `reference`（参考答案）为空，评测没法算指标；补上参考答案即可。体检不受影响。

## 上传与入库

- 上传返回 **413**：文件超过 `RAG_MAX_UPLOAD_MB`（默认 300）；经过反向代理时还要调大代理的 `client_max_body_size`。
- 上传返回 **409**：已有入库任务在运行。文件**已经保存**，等任务结束后调用 `POST /ingest` 即可。
- 上传返回 **400**：扩展名不支持、文件名无效（空 / 以 `.` 或 `~$` 开头 / `README.*`）或文件为空；`.doc` 请另存为 `.docx`。
- 任务 `failed`：看任务的 `error` 与 `skipped_files`，服务日志里有每个失败文件的 traceback；其余文件已正常入库。
- **服务运行时不要运行 `ragbase ingest --full`**（它会清空整个向量库，服务进程还持有旧索引），增量 `ragbase ingest` 也不要和服务同时跑。

## 其他

- **启动日志提示 `RAG_CHAT_MAX_SESSIONS` / `RAG_CHAT_SESSION_TTL` 已废弃**：这两项在会话持久化后不再生效（会话不自动淘汰），从 `.env` 里删掉即可，不影响启动。
- **启动时一大段 pydantic `ValidationError`**：某个配置值非法（枚举写错、数字超范围）。traceback 末尾会指出是哪个字段，如 `mode  Input should be 'vector', 'bm25' or 'hybrid'`。
- **端口被占用**：`ragbase serve --port 8001` 或设置 `RAG_PORT`。
- **依赖环境异常**：重新执行 `uv sync`（CI 用 `uv sync --locked`）。
- **Docker**：
  - Linux 上 `ragbase-home` 要由你自己先 `mkdir -p ragbase-home/data`（容器内是 uid 1000 的非 root 用户），否则容器里写不进去。
  - 镜像里没有模型：先 `docker compose run --rm ragbase models download`，或让 `up` 自带的 `--download-models` 自动下载。
  - Docker 相关文件还没有在真实环境里完成过镜像构建验证，遇到问题请以实际报错为准。
- **MPS 报错 / 不可用**：设置环境变量 `RAG_DEVICE=cpu`（默认 `auto` 会自动回落到 CUDA / CPU）。

还没解决？带上 `ragbase doctor` 的输出（已脱敏）和服务日志（`RAG_LOG_LEVEL=DEBUG` 可看更多）再排查。
