# HTTP API

服务由 FastAPI 实现（[`src/ragbase/api/app.py`](../src/ragbase/api/app.py)，请求 / 响应模型在 [`schemas.py`](../src/ragbase/api/schemas.py)）。启动：`uv run ragbase serve`，默认 <http://127.0.0.1:8000>。

**交互式文档**：服务运行时打开 `/docs`（Swagger UI）或 `/redoc`，`/openapi.json` 是机器可读的接口描述；它们总是由代码生成，字段细节以它们为准。

**必须单进程**：不要给 uvicorn 加 `--workers`。embedding / reranker / BM25 索引都加载在服务进程内，多 worker 会让每个进程各占一份显存，且 MPS 模型不能并发推理。并发请求由线程池接住，召回 + 精排在一把锁内串行，大模型生成在锁外并行。

## 目录

- [约定](#约定)：[鉴权](#鉴权)、[错误格式](#错误格式)、[请求 ID](#请求-id)
- [GET /health](#get-health)
- [POST /search](#post-search)、[POST /ask](#post-ask)
- [对话：POST /chat、POST /chat/stream、会话接口](#对话)
- [文档与入库：/documents、/ingest](#文档与入库)
- [题集：/eval-questions](#题集)
- [体检：/diagnose、报告](#体检)
- [前端路由与静态资源](#前端路由与静态资源)

## 约定

### 鉴权

设置 `RAG_API_KEY` 后，除下表的公开路径外，所有请求都必须带密钥；不设置（或留空）则不鉴权，适合本机使用。两种写法等价（同时带时任一匹配即可）：

```bash
curl -H "Authorization: Bearer $RAG_API_KEY" http://127.0.0.1:8000/documents
curl -H "X-API-Key: $RAG_API_KEY" http://127.0.0.1:8000/documents
```

缺少或错误的密钥返回 `401`，响应头带 `WWW-Authenticate: Bearer`，响应体是统一错误格式 `{"error": "unauthorized", "detail": "..."}`。鉴权在读取请求体**之前**完成：未鉴权的大文件上传和 `/chat/stream` 不会被处理。

**无需密钥的路径**（只限 `GET` / `HEAD`；注意 `/diagnose` 的 `GET` 是 Web 页面，同路径的 `POST /diagnose` 是体检接口，仍需密钥）：

| 路径 | 说明 |
| --- | --- |
| `/`、`/library`、`/diagnose`、`/c/{id}`、`/assets/*` | Web 界面页面与静态资源（里面没有数据，数据都走受保护的接口） |
| `/docs`、`/redoc`、`/openapi.json`、`/docs/oauth2-redirect` | 交互文档与接口描述（不含密钥） |
| `/health` | 没有有效密钥时只返回 `{"status": "ok" 或 "not_ready", "auth_required": true}`（存活探针 / 让前端知道要输入密钥）；带有效密钥返回完整内容；未启用鉴权时返回完整内容和 `"auth_required": false` |

鉴权是“**默认拒绝 + 显式公开白名单**”（[`api/auth.py`](../src/ragbase/api/auth.py)）：以后新增的任何路由默认都需要密钥，不会因为忘了加依赖而裸奔。命令行（`ragbase ask` / `ingest` 等）在进程内直接运行，不经过 HTTP，不需要密钥。密钥用常量时间比较，不会写进日志或错误响应。更多部署层面的建议见 [deployment.md](deployment.md#安全清单)。

**CORS**：默认不加 CORS 中间件（Web 界面与 API 同源托管，不需要）。前端部署在别的域名时设置 `RAG_CORS_ORIGINS=https://app.example.com,http://localhost:5173`（逗号分隔），服务会允许这些来源携带 `Authorization` / `X-API-Key` / `Content-Type` 头、任意方法访问，并暴露 `X-Request-ID`；不使用 cookie（`allow_credentials=False`）；预检请求不需要密钥。

### 错误格式

所有错误响应（包括 404、405）都是：

```json
{"error": "not_found", "detail": "没有这个会话：demo"}
```

`error` 是下表的机器码，`detail` 是中文说明（校验错误只列字段路径和原因，不回显你的输入；大模型 / 内部错误只给固定提示，完整异常写在服务日志里）。

| 状态码 | `error` | 何时出现 |
| --- | --- | --- |
| 400 | `bad_request` | 请求了未构建的 BM25 模式；`sources` 里有未知书名（`detail` 里列出可选项）；上传的文件名无效 / 格式不支持 / 文件为空；`POST /diagnose` 时题集为空 |
| 401 | `unauthorized` | 启用了鉴权但缺少或带了错误的密钥 |
| 404 | `not_found` | 没有这个文档 / 入库任务 / 会话 / 题目 / 体检任务 / 体检报告；未知路径 |
| 405 | `method_not_allowed` | 方法不对 |
| 409 | `conflict` | 已有入库任务、删除文档或体检在运行（三者共用一个“忙”状态，同一时刻只能有一个） |
| 413 | `payload_too_large` | 上传文件超过 `RAG_MAX_UPLOAD_MB` |
| 422 | `validation_error` | 参数校验失败（如 `query` 为空或超长、`session_id` 含非法字符、缺少上传字段 `file`）；题集接口的题目校验失败（响应多一个逐题错误列表 `errors`，见[题集](#题集)） |
| 502 | `llm_error` | 大模型调用失败（鉴权、限流、5xx 等，已按 `RAG_LLM_MAX_RETRIES` 重试过） |
| 503 | `service_unavailable` | 索引不存在或为空；大模型未配置（`/ask`、`POST /chat*`） |
| 504 | `llm_timeout` | 大模型响应超时（单次请求超过 `RAG_LLM_TIMEOUT` 且重试用完） |
| 500 | `internal_error` | 未处理的异常（详情只写日志）；对话时检索本身出错（模型加载、索引损坏等）也是它 |

其他未列出的状态码用 `http_error`。

### 请求 ID

每个响应带 `X-Request-ID`：请求头里带了合法值（`[A-Za-z0-9._-]{1,64}`）就沿用，否则生成。每个请求输出一行访问日志（方法、路径、状态码、耗时；`/search`、`/ask` 另带 `refused` 和 `top_score`，`/chat*` 带 `session`）。

### 索引为空时

索引不存在或为空时服务**仍会启动**：`/health` 报告 `not_ready`，`/search`、`/ask`、`/chat*`、`/documents*`、`/ingest` 返回 503（“索引不存在或为空，请先运行 ragbase ingest --full”）。用命令行建好索引后**需要重启服务**（在线入库要求服务启动时索引已存在）。

不依赖索引和大模型的接口照常可用：会话的列表、历史、删除（直接读写数据库，见[会话接口](#会话接口)），题集接口（`/eval-questions*`），体检报告接口（`GET`、`DELETE /diagnose/reports*`）。`POST /diagnose` 需要索引，索引为空返回 503（但不需要配置大模型）。

## GET /health

服务状态。索引和模型都就绪才是 `ok`。

| 字段 | 说明 |
| --- | --- |
| `status` | `ok` / `not_ready` |
| `chunk_count` | 索引片段数；索引不存在为 `null` |
| `bm25_ready` / `bm25_built_at` / `bm25_chunk_count` | BM25 是否可用、当前 BM25 索引的构建时间（UTC，ISO 8601）、覆盖的片段数（与 `chunk_count` 不同说明刷新未完成） |
| `device` | 本地模型的计算设备；embedding 与 reranker 都用云端时为 `cloud` |
| `embedding` / `reranker` | `{provider, model}`；`reranker` 在 `none` 时 `model` 为 `null` |
| `mode` | 默认召回方式 |
| `llm_configured` / `llm_protocol` / `llm_model` | 大模型是否配置齐全、协议、模型名（不含密钥） |
| `vector_store` | `chroma` / `qdrant` |
| `agent_ready` | 对话 Agent 是否就绪（大模型未配置或索引缺失时为 `false`） |
| `auth_required` | 是否启用了鉴权 |

Web 界面每 15 秒轮询它，用于右上角的状态指示。

## POST /search

召回 + 精排，**不调用大模型**。

请求：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `query` | string，必填 | 1 ~ 1000 字，首尾空白会被去掉 |
| `mode` | `vector` / `bm25` / `hybrid` | 默认取服务配置（`RAG_MODE`） |
| `top_k` | int，1 ~ 50 | 精排后返回的片段数，默认 `RAG_RERANK_K`（5） |
| `min_score` | number，0 ~ 1 | 拒答阈值，`0` 表示不拒答；默认 `RAG_MIN_SCORE` |
| `include_candidates` | bool，默认 `false` | 同时返回精排前的候选片段 `candidates` |
| `sources` | string[]，最多 50 项 | 按书过滤，见[按书过滤](#按书过滤sources) |

```bash
curl -s -X POST http://127.0.0.1:8000/search -H 'Content-Type: application/json' \
  -d '{"query": "关键路径法怎么计算总浮动时间", "mode": "hybrid", "top_k": 3}'
```

响应（数值仅示意）：

```json
{
  "refused": false,
  "top_score": 0.9996,
  "chunks": [
    {"rank": 1, "chunk_id": "…", "text": "…", "source": "xxx.pdf", "page": 333,
     "heading_path": "第11章 项目成本管理 > 11.3 挣值分析",
     "scores": {"vec_rank": 2, "bm25_rank": 2, "rrf_score": 0.0323, "rerank_score": 0.9996}}
  ],
  "took_ms": 3512.4
}
```

| 字段 | 说明 |
| --- | --- |
| `refused` | 最相关片段的得分低于 `min_score`，或没有召回（`chunks` 仍会返回，便于查看最接近的片段） |
| `top_score` | 最相关片段的精排得分；没有结果为 `null` |
| `chunks[]` / `candidates[]` | `rank`（在所在列表中的位置，1 起）、`chunk_id`、`text`、`source`（文件名，不含目录）、`page`（1 起，仅 PDF 有，否则 `null`）、`heading_path`（章节路径，旧索引或无书签的 PDF 为 `null`）、`scores` |
| `scores` | `vec_rank` / `bm25_rank`（该片段在向量路 / BM25 路内的排名，1 起，未命中为 `null`）、`rrf_score`（融合分数）、`rerank_score`（精排得分；精排前的候选和 `reranker=none` 时为 `null`） |
| `candidates` | 只有 `include_candidates=true` 时才出现 |
| `took_ms` | 服务端耗时（毫秒） |

## POST /ask

召回 + 精排 + 大模型回答。得分过低时**拒答且不调用大模型**。

请求：`query`（必填，1 ~ 1000 字）；可选 `mode`、`min_score`、`context_min_score`（只把得分不低于它的片段交给大模型，`0` 表示不过滤，默认 `RAG_CONTEXT_MIN_SCORE`）、`sources`。

```bash
curl -s -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"query": "关键路径法怎么计算总浮动时间"}'
```

响应（数值仅示意）：

```json
{
  "answer": "……[片段1]",
  "evidence": "- [片段1] “逐字摘录的原文”",
  "refused": false,
  "top_score": 0.9996,
  "citations": [{"n": 1, "chunk_id": "…", "text": "…", "source": "xxx.pdf", "page": 333,
                 "heading_path": "…", "scores": {"vec_rank": 2, "bm25_rank": 2, "rrf_score": 0.0323, "rerank_score": 0.9996}}],
  "took_ms": 6120.8
}
```

| 字段 | 说明 |
| --- | --- |
| `answer` | 最终回答（已去掉“依据”部分）。拒答时是固定的“我没有在文档中找到相关内容。” |
| `evidence` | 大模型逐字摘录的依据，没有则为空串（模型没按格式输出时整段当作回答、依据为空） |
| `refused` | 是否因得分过低 / 无召回拒答（未调用大模型） |
| `citations` | **`citations[n-1]` 对应回答里的 `[片段n]`**——它们就是实际交给大模型的片段（已按 `context_min_score` 过滤）；拒答时为空 |

大模型未配置返回 503；调用失败 502；超时 504。

## 对话

在 `/ask`（单轮、固定流程）之外，对话 Agent 由大模型自己决定是否检索、检索什么，再带 `[n]` 引用作答；支持多轮追问。机制见 [architecture.md](architecture.md#对话-agent)。

### POST /chat

请求：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `session_id` | string，可选 | 字母 / 数字 / `_` / `-`，最长 64；不传则新建并在响应里返回。同一个 `session_id` 共享历史，同会话的请求串行处理 |
| `message` | string，必填 | 1 ~ 2000 字，首尾空白会被去掉 |
| `sources` | string[]，可选 | 作为本轮检索的**默认**书目过滤；模型自己在工具调用里显式传了 `sources` 则以模型为准 |

响应：

```json
{
  "session_id": "demo",
  "answer": "进度绩效指数 SPI = EV / PV …… [1]",
  "citations": [{"n": 1, "chunk_id": "…", "text": "…", "source": "xxx.pdf", "page": 333, "heading_path": "…", "scores": {"…": "…"}}],
  "tool_calls": [{"name": "search_knowledge", "args": {"query": "挣值分析中进度绩效指数 SPI 的计算公式是什么？"}}],
  "took_ms": 5234.1
}
```

- `tool_calls`：本轮模型调用过的工具（按顺序）。工具有 `search_knowledge`（`query`、可选 `sources`）与 `list_documents`。
- `citations`：见下面的[引用编号](#引用编号语义)。

Agent 未就绪（大模型未配置或索引缺失）返回 503；调用失败 502；超时 504。失败的那一轮会回滚，会话保持可用。

```bash
curl -s -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' \
  -d '{"session_id": "demo", "message": "什么是挣值分析？"}'
curl -s -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' \
  -d '{"session_id": "demo", "message": "它的进度绩效指数怎么算？"}'      # 追问：同一个 session_id
```

### POST /chat/stream

请求体同 `/chat`，以 Server-Sent Events 流式返回（`Content-Type: text/event-stream`，响应头 `Cache-Control: no-cache`、`X-Accel-Buffering: no`）。**校验类错误（400 / 401 / 422 / 503）在开始流式之前以普通 JSON 返回**；流式过程中的错误以 `error` 事件发出。

事件顺序：`session` → （`tool_call` 与 `token` 交错出现）→ `citations` → `done`；出错时在中途以 `error` 结束。

| 事件 | `data`（JSON） | 说明 |
| --- | --- | --- |
| `session` | `{"session_id": "…"}` | 总是第一个事件 |
| `tool_call` | `{"name": "search_knowledge", "args": {…}}` | 模型决定调用工具时发出 |
| `token` | `{"text": "…"}` | 模型每次生成的文本增量 |
| `citations` | `{"citations": [Citation, …]}` | 回答里实际标注了 `[n]` 的片段 |
| `done` | `{"answer": "…", "took_ms": 5234.1}` | **`answer` 是权威的最终回答** |
| `error` | `{"error": "llm_timeout", "detail": "…"}` | 同统一错误 JSON：`llm_timeout` / `llm_error` / `internal_error`；流到此结束 |

注意：若模型在调用工具前先说了一句话，这句话也会出现在 `token` 流里，所以请以 `done.answer` 为准。没有逐字流出时（例如触达模型调用上限，由中间件直接给出提示）会把整段回答作为一个 `token` 补发。客户端中途断开会让本轮回滚。

一个完整的流（数值与内容仅示意，`data` 在线上是单行 JSON）：

```text
event: session
data: {"session_id": "demo"}

event: tool_call
data: {"name": "search_knowledge", "args": {"query": "挣值分析中进度绩效指数 SPI 的计算公式是什么？"}}

event: token
data: {"text": "SPI = EV / PV"}

event: token
data: {"text": "，用来衡量进度绩效 [1]。"}

event: citations
data: {"citations": [{"n": 1, "chunk_id": "…", "text": "…", "source": "xxx.pdf", "page": 333, "heading_path": "…", "scores": {"vec_rank": 1, "bm25_rank": 2, "rrf_score": 0.0325, "rerank_score": 0.9998}}]}

event: done
data: {"answer": "SPI = EV / PV，用来衡量进度绩效 [1]。", "took_ms": 5234.1}
```

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream -H 'Content-Type: application/json' \
  -d '{"session_id": "demo", "message": "再举个例子"}'      # -N 关闭 curl 缓冲
```

经过反向代理时必须关闭对该路径的响应缓冲，见 [deployment.md](deployment.md#反向代理与-https)。

### 会话接口

会话保存在内置 SQLite 数据库里（`RAG_HOME/ragbase.db`，见[架构：数据库](architecture.md#数据库)）：**重启服务不丢**，每轮问答、引用编号和引用片段都完整保留（Web 上能翻看整段对话，只是发给大模型时截取最近 `RAG_CHAT_HISTORY_TURNS` 轮）。会话**不会自动清理**，没有数量上限，也不会因为空闲过期——只有手动删除（下表的 `DELETE`，或 Web 上的删除）才会清掉。`RAG_CHAT_MAX_SESSIONS` 和 `RAG_CHAT_SESSION_TTL` 已废弃，设置了也不再生效。

会话列表、历史和删除**不依赖对话 Agent**：大模型未配置或索引为空（`/health` 的 `agent_ready` 为 `false`）时，它们照常从数据库读写，只有发消息（`POST /chat`、`POST /chat/stream`）返回 503。

| 接口 | 响应 |
| --- | --- |
| `GET /chat/sessions` | `{"sessions": [{"session_id", "title", "updated_at", "message_count"}]}`，按最近使用倒序，没有分页；`title` 是第一条用户消息（单行，最多 30 字，超出加 `…`），第一轮成功后固定、之后不变；`message_count` 累计用户消息与最终回答（不含工具调用步骤）；第一轮没有成功的会话不会出现在列表里 |
| `GET /chat/sessions/{id}` | `{"session_id", "messages": [...]}`，每条消息 `role`（`user` / `assistant` / `tool`）+ `content`；助手消息可带 `tool_calls`，并带 `citations`（该条正文里标注了 `[n]` 的片段）；工具消息的 `content` 只给前 80 字的摘要。返回会话的**全部**消息，不受历史窗口影响。会话不存在（含已被删除）返回 404 |
| `DELETE /chat/sessions/{id}` | `{"session_id", "deleted": true}`；会话不存在返回 404。删除该会话的消息历史、引用和会话记录；对话 Agent 在线时，等该会话正在处理的请求结束后执行 |

```bash
curl -s http://127.0.0.1:8000/chat/sessions/demo
curl -s -X DELETE http://127.0.0.1:8000/chat/sessions/demo
```

### 引用编号语义

`/chat*` 的引用编号是**会话级**的（`/ask` 则是单次请求内的 `[片段n]`）：

- 同一个会话里，**同一个片段永远是同一个编号**，后续轮次检索到的新片段接着往下编；编号不会因为历史被裁剪而变化。
- 回答里用 `[n]` 标注；响应的 `citations` 只含**回答里实际标注了 `[n]` 的片段**，所以 `n` 可能不连续，也可能包含更早几轮登记的片段（模型沿用历史里出现过的编号作答时同样能解析）。回答里出现了从未登记过的编号会被丢弃，不会出现在 `citations` 里。
- 编号随会话保存在数据库里：**重启服务后接着追问，编号继续接着之前的走，同一片段的编号不变**。
- 一轮失败 / 中断时，该轮新登记的片段会被撤销（服务进程在一轮中途被强杀时，下一轮开始时会把这半截轮次整轮清掉，引用同样撤销）；会话被删除时编号随之丢弃。
- 因此 `GET /chat/sessions/{id}` 里的助手消息可以带着它当时的引用，Web 界面重新打开历史会话也能渲染引用角标和“依据”面板。

## 文档与入库

文档管理在服务进程内完成：入库任务在后台线程里跑，**同一时刻只有一个**（入库和删除文档共用这个“忙”状态）。这些接口要求服务启动时索引非空（否则 503）。

| 接口 | 说明 |
| --- | --- |
| `GET /documents` | `{"documents": [{"name", "chunk_count", "heading_chunks", "has_headings"}]}`，按文件名汇总（旧版索引同样可用）。`name` 可用于 `sources` 过滤和 `DELETE` |
| `POST /documents` | 上传文档（`multipart/form-data`，字段名 **`file`**），保存到数据目录并立即开始增量入库，返回 `202` + 任务信息（另含 `filename`，实际保存的文件名） |
| `DELETE /documents/{name}` | 删除文档：源文件、索引片段、入库记录，并同步刷新 BM25；返回 `{"name", "chunks_deleted"}`。没有这个文档返回 404；有任务在运行返回 409 |
| `POST /ingest` | 对数据目录做一次增量入库（新增 / 修改 / 删除的文件），返回 `202` + 任务信息；已有任务在运行返回 409 |
| `GET /ingest/jobs/{job_id}` | 任务状态；不存在返回 404（内存里最多保留最近 100 个任务） |

```bash
curl -s -X POST http://127.0.0.1:8000/documents -F 'file=@新教材.pdf'    # 202 {"job_id": "…", "status": "running", "filename": "新教材.pdf", …}
curl -s http://127.0.0.1:8000/ingest/jobs/<job_id>                      # 轮询：pending | running | succeeded | failed
curl -s -X POST http://127.0.0.1:8000/ingest                            # 不上传，只对数据目录重新做一次增量入库
curl -s -X DELETE 'http://127.0.0.1:8000/documents/新教材.pdf'
```

**任务状态字段**：`job_id`、`status`（`pending` / `running` / `succeeded` / `failed`）、`files_total`、`files_done`、`current_file`、`chunks_added`、`chunks_deleted`、`skipped_files`（被跳过的旧版 `.doc` 和入库失败的文件）、`error`、`started_at`、`finished_at`（UTC，ISO 8601）。`succeeded` 表示入库完成**且 BM25 已刷新**；有文件失败时整体 `failed`，其余文件已正常入库并写入记录。

**上传校验**（`POST /documents`）：

- 扩展名（不区分大小写）：`.pdf` `.docx` `.md` `.markdown` `.txt`；`.doc` 会提示另存为 `.docx`；其他返回 400。
- 大小：`RAG_MAX_UPLOAD_MB`（默认 300），流式写盘不整体读入内存，超限 413；`Content-Length` 明显超限时不等请求体传完就拒绝。
- 文件名只取最后一段（`../x.pdf` 会存成 `x.pdf`）；空、以 `.` 或 `~$` 开头、文件名主干为 `README`（如 `README.md`）、含控制字符、过长的名字返回 400；空文件返回 400。
- 同名文件覆盖（原子替换）。已有任务在运行时文件**仍会保存**，但返回 409——稍后调用 `POST /ingest` 即可。

**并发**：入库每 64 个片段为一批（向量化 + 写入）单独持有检索锁，批间释放，所以入库期间 `/search` 可以插进来（会比平时慢一点）。同名修改的文件先写入新片段、再删不再存在的旧片段，重建期间该文件仍可检索。任务结束后重建 BM25 快照并原子替换。

## 题集

评测题集（体检和 `ragbase eval` 共用）存在内置数据库里，用这组接口管理；题集本身的概念、字段与导入规则见[评测题格式](evaluation.md#评测题格式)。接口不依赖对话 Agent，大模型未配置也能用。

| 接口 | 说明 |
| --- | --- |
| `GET /eval-questions` | `{"items": [题目…]}`，按 `id` 升序 |
| `POST /eval-questions` | 新增一道题，`id` 自动分配；`201` 返回该题 |
| `PUT /eval-questions/{id}` | 只改请求里出现的字段，传 `null` 清空可选字段；返回该题。没有这道题返回 404 |
| `DELETE /eval-questions/{id}` | `{"deleted": true}`；没有这道题返回 404 |
| `POST /eval-questions/import` | 批量导入，见下 |
| `GET /eval-questions/export` | 整个题集，JSON 数组，响应头 `Content-Disposition: attachment; filename="eval_questions.json"`；每题同时有 `reference` 和 `answer_keyword`（取值相同），旧版 `ragbase eval` 也能读 |

**题目**：`id`、`question`、`reference`（参考答案，可为 `null`）、`difficulty`、`type`、`source_topic`（均可为 `null`）、`created_at`、`updated_at`（服务器本地时间，ISO 8601）。新增 / 修改时请求体可带 `question`（新增必填，非空字符串）、`reference`、`difficulty`、`type`、`source_topic`，也接受旧字段名 `answer_keyword`（等同 `reference`，两个都给时取 `reference`）。

**导入**：请求体 `{"items": [...], "mode": "append" | "replace"}`（`mode` 默认 `append`；`items` 是题目数组，与导出文件同格式）。成功返回 `{"imported": 30, "reassigned": [[旧 id, 新 id], …]}`：`append` 时 `id` 与已有题冲突的题会改用新 `id`，记在 `reassigned` 里；`replace` 先清空题集再按文件里的 `id` 写入。校验和写入在同一个事务里，**任何一题不合法都整体不写入**，返回 422。

```bash
curl -s http://127.0.0.1:8000/eval-questions
curl -s -X POST http://127.0.0.1:8000/eval-questions -H 'Content-Type: application/json' \
  -d '{"question": "什么是关键路径？", "reference": "最长的活动路径，决定项目最短工期"}'
curl -s -X POST http://127.0.0.1:8000/eval-questions/import -H 'Content-Type: application/json' \
  -d '{"items": [{"id": 1, "question": "什么是软件架构风格？", "answer_keyword": "惯用模式"}], "mode": "append"}'
curl -s -o eval_questions.json http://127.0.0.1:8000/eval-questions/export
```

**校验失败（422）**：在统一错误格式 `{error, detail}` 之外多一个 `errors` 数组，每项 `{index, field, message}`：`index` 是出错题目在导入数组里的下标（**从 0 开始**）；整体错误（如输入不是数组）以及新增 / 修改时为 `null`；`field` 是出错字段，整体错误时为 `null`；`message` 是中文说明。例如导入 `{"items": [{"question": "a"}, {"id": "x"}], "mode": "append"}`：

```json
{
  "error": "validation_error",
  "detail": "题集校验未通过（共 2 处）：第 2 题（question）缺少必填字段 question（非空字符串）；第 2 题（id）必须是正整数（实际是 'x'）",
  "errors": [
    {"index": 1, "field": "question", "message": "缺少必填字段 question（非空字符串）"},
    {"index": 1, "field": "id", "message": "必须是正整数（实际是 'x'）"}
  ]
}
```

`detail` 最多列出前 3 处，完整列表看 `errors`。请求体本身不是合法 JSON、`mode` 不是 `append` / `replace`、路径里的 `id` 不是整数等，则是普通的 422（没有 `errors`）。

## 体检

体检（`ragbase diagnose` 的 Web 版）在服务进程的后台线程里运行，**不调用大模型**，复用服务已加载的模型；每份报告存进数据库、保留到手动删除。报告各项的含义见 [diagnostics.md](diagnostics.md)。

| 接口 | 说明 |
| --- | --- |
| `POST /diagnose` | 开始一次体检，返回 `202` + 任务信息。请求体可选：`mode`（`vector` / `bm25` / `hybrid`，默认服务配置）、`limit`（整数 ≥ 1，只用题集前 N 道）。无请求体也可以 |
| `GET /diagnose/jobs/{job_id}` | 任务状态；不存在返回 404（内存里最多保留最近 100 个任务，服务重启后任务记录消失，但已生成的报告还在） |
| `GET /diagnose/reports` | `{"items": [报告摘要…]}`，按 `id` 倒序（新的在前）；不含逐题记录 |
| `GET /diagnose/reports/{id}` | 完整报告；没有这份报告返回 404 |
| `DELETE /diagnose/reports/{id}` | `{"deleted": true}`；没有这份报告返回 404 |

`POST /diagnose` 的错误：

- `400`：题集为空（先添加或导入题目）；请求了服务未构建的 BM25 模式；
- `409`：已有体检、入库任务或删除文档在运行（体检与它们互斥；反过来，体检运行时发起 `POST /ingest`、`POST /documents`、`DELETE /documents/{name}` 也返回 409）；
- `503`：索引不存在或为空。

题目在返回 `202` 之前就取好快照（先按 `limit` 截取、再对同文本重复的问题去重），运行期间编辑或删除题目不影响这次体检。体检期间对话请求能插进来，但会变慢；报告里记录了期间服务处理的对话请求数。

**任务状态字段**：`job_id`、`status`（`pending` / `running` / `succeeded` / `failed`）、`total`（本次体检的题数，去重、按 `limit` 截取之后）、`done`（已测完的题数）、`report_id`（成功后的报告 id，否则为 `null`）、`error`、`started_at`、`finished_at`（UTC，ISO 8601）。

**报告摘要**（列表项）：`id`、`created_at`（服务器本地时间，ISO 8601）、`origin`（`cli` / `web`）、`mode`、`question_count`、`candidates_recall`（召回 TopK 的 `point_recall` 均值）、`reranked_recall`（精排后）、`refusal_rate`、`total_ms_mean`（检索总耗时均值，毫秒）、`rerank_share_pct`（精排占比），后面几个数字在报告缺少对应数据时为 `null`；另有 `concurrent_chat_requests`（体检期间的对话请求数，大于 0 时耗时可能偏高）。

**完整报告**：`id`、`origin`、`generated_at`、`params`、`warmup_question`、`questions`（逐题记录）、`summary`、`recommendations`、`question_ids`、`question_digest`（题集内容摘要，两份报告不同说明题集不同）、`concurrent_chat_requests`。结构说明见[体检：导出与保留](diagnostics.md#导出与保留)；字段细节以 `/docs` 为准。

```bash
curl -s -X POST http://127.0.0.1:8000/diagnose -H 'Content-Type: application/json' -d '{"mode": "hybrid", "limit": 10}'
# 202 {"job_id": "…", "status": "running", "total": 10, "done": 0, "report_id": null, …}
curl -s http://127.0.0.1:8000/diagnose/jobs/<job_id>        # 轮询：done 递增，成功后 report_id 有值
curl -s http://127.0.0.1:8000/diagnose/reports              # 报告列表（命令行跑出来的报告也在里面）
curl -s http://127.0.0.1:8000/diagnose/reports/3
curl -s -X DELETE http://127.0.0.1:8000/diagnose/reports/3
```

### 按书过滤（`sources`）

`/search`、`/ask`、`/chat*` 的 `sources` 是书名（文件名，取自 `GET /documents` 的 `name`）列表，只在这些书里检索；缺省或空列表表示全部；有未知书名返回 400 并列出可选项（对话里则把可选项告诉模型让它自己纠正）。命令行：`ragbase ask --source 书名.pdf`（可重复）。

```bash
curl -s -X POST http://127.0.0.1:8000/search -H 'Content-Type: application/json' \
  -d '{"query": "挣值分析", "sources": ["某某教程.pdf"]}'
```

过滤后没有命中则判定拒答。

## 前端路由与静态资源

`web/dist`（或随包的前端副本）存在时，API 服务在同一端口提供：`/assets/*`（带内容哈希，长缓存）和前端路由 `/`、`/library`、`/diagnose`、`/c/{session_id}`（返回 `index.html`）。这些路径不会遮住 `GET /documents`、`/chat/sessions`、`/health` 等 JSON 接口；`/diagnose` 只有 `GET` 是页面，`POST /diagnose` 仍是体检接口。目录是否存在在请求时检查，服务运行中重新构建前端立即生效；未构建时 `GET /` 返回提示“前端未构建”。
