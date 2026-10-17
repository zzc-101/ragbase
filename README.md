# RAG Demo

面向中文 PDF、Word、Markdown 和文本资料的本地 RAG 命令行问答 Demo。

## 架构

```text
文档 → 解析 → 清洗 → 中文切分 → Qwen3-Embedding-0.6B → ChromaDB
                                                      ↓
提问 → 向量召回 + BM25 召回 → RRF 融合 Top 20 → Qwen3-Reranker 精排 Top 5 → 豆包生成 → 答案与来源
```

- Embedding 和 Reranker 在本机运行，设备自动选择（Apple Silicon 的 MPS > CUDA > CPU，可用 `RAG_DEVICE` 指定）。
- 两个本地模型会优先从本机缓存加载；重复问题的查询向量和精排分数会在进程内缓存。
- 回答模型通过 OpenAI 兼容接口调用火山引擎方舟。
- 向量库、文档和密钥均已在 `.gitignore` 中排除。

入库预处理按职责拆分为：

- `document_parser.py`：文件格式解析
- `document_cleaner.py`：文本规范化、重复页眉页脚和空页清理
- `document_splitter.py`：递归切分、空片段与重复片段过滤
- `document_toc.py`：PDF 书签 -> 每页章节路径（`heading_path`）
- `rag_common.py`：稳定 ID、文件指纹和入库编排
- `ingest_service.py`：增量入库逻辑（命令行和 HTTP 服务共用，依赖全部显式传入，便于用临时目录 + 假 embedding 测试）

## 混合检索

[hybrid_retriever.py](hybrid_retriever.py) 同时走两路召回：向量路直接查 Chroma（各取 30），BM25 路用 `jieba` 分词 + `bm25s` 打分（各取 30），再用 RRF（`score = Σ 1/(60+rank)`，只看排名）融合取前 20 交给 Reranker。专有名词、缩写和公式（如 TOGAF、SPI）这类向量容易漏的内容，由 BM25 补召回。

- BM25 索引在启动时从 Chroma 已有的片段文本现建，无需改动入库流程。
- 分词结果缓存在 `.bm25_cache/`（已忽略）；缓存键包含全部片段内容的摘要和用户词典，入库 / 删除 / 章节标题变化后自动失效，旧缓存文件会被清掉。
- BM25 索引、语料和书目录打包成不可变的“快照”，入库或删除文档后在后台建好新快照，再整体替换一个引用（原子切换），检索期间不会看到半成品索引。
- 领域词典放在仓库根目录的 [bm25_userdict.txt](bm25_userdict.txt)（jieba 格式：`词 词频 词性`）。不要放进 `data/`，那里的 `.txt` 会被当作文档入库。修改后下次启动自动生效。
- 召回模式用 `--mode` 切换：`uv run query.py --mode {vector,bm25,hybrid}`（默认 `hybrid`）。答案后的来源会标出召回路径，如 `[V3+B1]` 表示向量路第 3、BM25 路第 1。
- 单元测试：`uv run python tests/test_hybrid.py`。

## 章节标题（PDF 书签）

入库时读取 PDF 书签（目录），给每个片段加上 `heading_path`（如 `第11章 项目成本管理 > 11.3 挣值分析`），见 [document_toc.py](document_toc.py)：

- 对第 p 页，路径 = 起始页不晚于该页的各层级“最近一条”书签标题，用 ` > ` 连接；标题会去首尾空白、压缩空白，纯数字 / 标点的标题（如考试大纲里的 `1`）直接丢弃，没有可用书签则该文件没有章节。非 PDF 暂无章节。
- **近似**：书签只记录起始页，不记录页内位置，一节从页中间开始时，整页都归给这一节。
- 向量化输入 = `heading_path + "\n" + 正文`，BM25 分词输入 = `heading_path + " " + 正文`；Chroma 里存的 document 仍是纯正文。**精排 Reranker 的输入只用正文**，这样按纯正文校准的拒答阈值仍然有效。
- 展示：`/search` 的 `chunks`、`/ask` 的 `citations` 带 `heading_path`（没有为 `null`）；交给大模型的上下文写成 `[片段1]（第11章 … > 11.3 …）正文`；`query.py` 的来源标签带末级章节，如 `xxx.pdf:第12页「11.3 挣值分析」[V2+B1|0.93]`。
- `INGEST_PIPELINE_VERSION` 已升到 `3`、分词版本升到 `3`：下次增量入库会把所有文件视为已变化并重新索引（这才会给现有的 PDF 加上章节）。**旧索引（没有 `heading_path`）无需改动即可继续使用**，章节全部为 `null`。

## 低相关度拒答

精排后，如果最相关片段的 `rerank_score` 低于阈值，直接回答“我没有在文档中找到相关内容。”，**不调用大模型**（省钱，也避免模型凭空编造）。

- `rerank_score` 是 Reranker 给出的 P(相关)，范围 0~1（模型原始输出是 logit 差，经 sigmoid 转换）。每个返回片段的 metadata 里都有，命令行来源会显示成 `xxx.pdf:第12页[V2+B1|0.93]`。
- 默认阈值是 [config.py](config.py) 的 `RERANK_MIN_SCORE`（也可用 `RAG_MIN_SCORE` 环境变量覆盖）；临时调整用 `--min-score`，`0` 表示不拒答：

```bash
uv run query.py --min-score 0.3
uv run evaluate.py --retrieval-only --min-score 0.3   # 同时统计会被拒答的题数
```

- 重新校准（换了教材、Reranker 或召回方式后）：运行下面的脚本，不调用 LLM。它会对 30 道评测题和一组库外问题打印最相关片段的得分分布，以及各候选阈值的“误拒 / 漏拦”数，再把推荐值填回 `RERANK_MIN_SCORE`。需要拦截更多库外问题时可以调高阈值，代价是更容易误拒正常问题。

```bash
uv run python scripts/calibrate_rerank_threshold.py
```

- 单元测试：`uv run python tests/test_refusal.py`。

## 抑制幻觉：片段过滤与“先摘录再回答”

**1. 只把足够相关的片段交给大模型。** 通过拒答门槛后，只保留 `rerank_score >= CONTEXT_MIN_SCORE` 的片段（最相关的 top1 总是保留），其余不进入提示词。默认值 `0.95` 定义在 [config.py](config.py)（环境变量 `RAG_CONTEXT_MIN_SCORE`），旁边的注释记录了 30 道评测题的阈值实验。目前 Reranker 得分很饱和（库内题 Top5 的得分基本都 ≥ 0.96），所以该阈值对库内题几乎不裁剪，主要是兜底；调高会开始丢掉对比 / 多方面题需要的片段。临时调整：

```bash
uv run query.py --context-min-score 0.99      # 0 表示不过滤
uv run evaluate.py --retrieval-only --context-min-score 0.99
```

`ask()` 返回 `AskResult`（`answer / evidence / docs / reranked / candidates / refused / top_score`），其中 `docs` 就是实际交给大模型的片段，`[片段N]` 与 `docs` 的顺序一一对应。

**2. 先逐字摘录依据，再作答。** 提示词要求模型按下面的格式输出，`split_evidence_answer()` 负责拆开；模型没按格式输出时，整段当作回答：

```text
【依据】
- [片段N] “逐字摘录的原文”
【回答】
基于依据的回答，句末标注 [片段N]
```

命令行依次打印“依据”“答”和与 `[片段N]` 对应的来源列表；评测只用【回答】部分做 Ragas / 裁判，依据单独存进结果的 `evidence` 字段。评测结果里 `lexical.llm_context` 与 `llm_contexts` 是大模型实际看到的片段，原有的 `reranked_top5` / `retrieved_contexts`（未过滤的精排 Top5）保持不变。

- 单元测试：`uv run python tests/test_context.py`。

## 环境准备

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

```bash
cd /Users/zc/MyProject/ragdemo
uv venv --python 3.12
uv pip sync requirements.lock
```

创建 `.env`：

```dotenv
RAG_LLM_API_KEY=你的 API Key
RAG_LLM_MODEL=模型名（方舟填推理接入点 ID）
RAG_LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
CHROMA_TELEMETRY_DISABLED=true
```

大模型走 OpenAI 兼容协议，可换成任意兼容的服务商，详见“服务化运行”里的“大模型配置”。旧的 `ARK_CHAT_API_KEY` / `ARK_CHAT_MODEL` / `ARK_BASE_URL` 仍然有效。

## 建立索引

将资料放入 `data/`，支持：

- `.pdf`
- `.docx`
- `.md` / `.markdown`
- `.txt`

只扫描 `data/` 顶层（不递归子目录），扩展名不区分大小写；`README.*` 不会被当作文档。旧版 `.doc` 无法解析，会被跳过并提示，请先另存为 `.docx`。

首次使用或需要彻底重建时：

```bash
uv run ingest.py
```

以后文档新增、修改或删除时：

```bash
uv run ingest_incremental.py
```

全量入库和增量入库使用相同的稳定片段 ID，并通过 `.ingest_meta.json` 记录文件状态（每个文件成功后立即写入，中途失败不会重做已完成的文件；失败的文件下次重试）。旧版索引首次执行增量命令时会迁移重建：先按 ID 分批清空旧片段，再逐文件写入（不删除 collection）。

> **服务运行期间不要运行 `ingest.py`**：它会删除整个 `chroma_db/` 目录，而服务进程还持有该索引。服务在线时请用下面的 `POST /documents` / `POST /ingest` / `DELETE /documents/{name}`（或先停服务再跑命令行）。`ingest_incremental.py` 同样不要与服务同时运行（两个进程会同时写同一个库）。

## 问答与评测

启动交互式问答：

```bash
uv run query.py
```

输入 `exit`、`quit` 或 `q` 退出。

运行 30 道内置评测题：

```bash
uv run evaluate.py
```

评测包含三层指标：

- Ragas：忠实度、回答相关性、事实正确性、上下文精确率和召回率
- 无需 LLM 的词面代理指标：要点召回、上下文精确率和 MRR（分别统计召回 Top20、精排 Top5 和实际交给 LLM 的片段）
- 原有中文 LLM 裁判：回答正确性与检索充分性

先用一道题冒烟测试：

```bash
uv run evaluate.py --limit 1
```

只运行词面指标和问答生成、跳过额外的评测模型调用：

```bash
uv run evaluate.py --skip-ragas --skip-legacy-judge
```

对比召回方式（`--retrieval-mode {vector,bm25,hybrid}`，默认 `hybrid`），结果写入 `eval_results.<mode>.json`。加上 `--retrieval-only` 则只跑召回 + 精排和词面指标，不调用任何 LLM，没有 API 费用（结果写入 `eval_results.<mode>.retrieval.json`）：

```bash
uv run evaluate.py --retrieval-only --retrieval-mode vector
uv run evaluate.py --retrieval-only --retrieval-mode hybrid
```

结果逐题写入对应的结果文件，Ragas 调用缓存于 `.ragas_cache/`。评测会调用回答模型，因此会产生 API 费用。当前测试集的 `answer_keyword` 会作为参考答案；后续可以为题目增加更完整的 `reference` 字段以提高评测可信度。

> 词面指标是基于参考答案要点的代理指标，不是基于人工相关文档标注的标准 Precision/Recall/MRR。

## 服务化运行

同一份代码也可以作为 HTTP 服务运行（FastAPI，[api.py](api.py)），`query.py` / `evaluate.py` 仍可独立使用。

```bash
uv run uvicorn api:app --host 127.0.0.1 --port 8000   # 或：uv run api.py（读取 RAG_HOST / RAG_PORT）
```

- **必须单进程，不要加 `--workers`**：embedding / reranker / BM25 索引都加载在服务进程内，多 worker 会让每个进程各占一份显存，且 MPS 模型不能并发推理。并发请求由线程池接住，召回 + 精排在一把锁内串行，大模型生成在锁外并行（慢生成不会阻塞检索）。
- 启动约 10 秒（加载两个本地模型和 BM25 索引）。索引不存在或为空时服务仍会启动，`/health` 报告 `not_ready`，`/search`、`/ask` 以及文档管理接口返回 503（“索引不存在或为空，请先运行 uv run ingest.py”）；用命令行建好索引后需重启服务（在线入库要求服务启动时索引已存在）。
- 每个响应带 `X-Request-ID`（请求头里带了就沿用，否则生成），每个请求输出一行访问日志（方法、路径、状态码、耗时；`/search`、`/ask` 另带 `refused` 和 `top_score`）。
- 错误统一为 `{"error": "...", "detail": "..."}`：422 参数校验、400 请求了未构建的 BM25 模式 / 未知书名 / 非法上传、404 找不到文档或任务、409 已有入库任务在运行、413 上传文件过大、502 大模型调用失败（鉴权、限流、5xx 等，已按 `RAG_LLM_MAX_RETRIES` 重试过）、504 大模型响应超时（`error` 为 `llm_timeout`，单次请求超过 `RAG_LLM_TIMEOUT` 且重试用完）、503 索引 / 大模型未就绪、500 未处理异常（详情只写日志）。
- 交互式文档：<http://127.0.0.1:8000/docs>。

| 接口 | 说明 |
| --- | --- |
| `GET /health` | `status`（`ok` / `not_ready`）、`chunk_count`、`bm25_ready`、`bm25_built_at`（当前 BM25 索引的构建时间，UTC）、`bm25_chunk_count`（BM25 覆盖的片段数，与 `chunk_count` 不同说明刷新未完成）、`device`、`mode`、`llm_configured`、`llm_protocol`、`llm_model`（模型名，不含密钥）、`agent_ready`（对话 Agent 是否就绪） |
| `POST /search` | 召回 + 精排，不调用大模型 |
| `POST /ask` | 召回 + 精排 + 大模型回答；得分过低时拒答且不调用大模型 |
| `GET /documents` | 已入库的文档：`name`、`chunk_count`、`heading_chunks`、`has_headings`（按 `source` 的文件名汇总，旧版索引同样可用） |
| `POST /documents` | 上传文档（multipart，字段名 `file`），保存到数据目录并立即开始增量入库，返回 202 + 任务信息 |
| `DELETE /documents/{name}` | 删除文档：源文件、索引片段、元数据，并同步刷新 BM25 |
| `POST /ingest` | 对数据目录做一次增量入库，返回 202 + `job_id` |
| `GET /ingest/jobs/{job_id}` | 任务状态 |

`/search` 请求：`query`（1~1000 字，首尾空白会被去掉）必填；可选 `mode`（`vector` / `bm25` / `hybrid`）、`top_k`（精排后返回数，默认 5）、`min_score`（0 表示不拒答）、`include_candidates`（同时返回精排前的候选，默认 false）、`sources`（见下“按书过滤”）。

```bash
curl -s -X POST http://127.0.0.1:8000/search -H 'Content-Type: application/json' \
  -d '{"query": "关键路径法怎么计算总浮动时间", "mode": "hybrid", "top_k": 3}'
```

示例响应（数值仅示意）：

```json
{
  "refused": false,
  "top_score": 0.9996,
  "chunks": [
    {"rank": 1, "chunk_id": "...", "text": "...", "source": "xxx.pdf", "page": 333,
     "heading_path": "第11章 项目成本管理 > 11.3 挣值分析",
     "scores": {"vec_rank": 2, "bm25_rank": 2, "rrf_score": 0.0323, "rerank_score": 0.9996}}
  ],
  "took_ms": 3512.4
}
```

`refused` 为 true 表示最相关片段得分低于 `min_score`（`chunks` 仍会返回，便于查看）；`page` 为 1 起的页码，仅 PDF 有，否则为 null；`heading_path` 是章节路径，旧版索引或没有书签的 PDF 为 null。

`/ask` 请求：`query` 必填；可选 `mode`、`min_score`、`context_min_score`、`sources`。

```bash
curl -s -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"query": "关键路径法怎么计算总浮动时间"}'
```

示例响应（数值仅示意）：

```json
{
  "answer": "……[片段1]",
  "evidence": "- [片段1] “逐字摘录的原文”",
  "refused": false,
  "top_score": 0.9996,
  "citations": [{"n": 1, "chunk_id": "...", "source": "xxx.pdf", "page": 333, "text": "...", "scores": {"...": "..."}}],
  "took_ms": 6120.8
}
```

`citations[n-1]` 对应回答中的 `[片段n]`（含 `heading_path`）；拒答时 `citations` 为空。

**按书过滤**：`/search` 与 `/ask` 的 `sources` 是书名（文件名，取自 `GET /documents` 的 `name`）列表，只在这些书里检索；缺省或空列表表示全部；有未知书名返回 400 并列出可选项。服务端用检索器内存里的语料把书名解析成库里存的 `source` 原始值（旧索引存的是绝对路径，不会自己拼路径）；向量路用 Chroma 的 `where={"source": {"$in": [...]}}`，BM25 路用 bm25s 的 `weight_mask`，过滤后没有命中则判定拒答。命令行：`uv run query.py --source 书名.pdf`（可重复）。

```bash
curl -s -X POST http://127.0.0.1:8000/search -H 'Content-Type: application/json' \
  -d '{"query": "挣值分析", "sources": ["某某教程.pdf"]}'
```

**在线入库**（任务在服务进程的后台线程里跑，同一时刻只有一个）：

```bash
curl -s -X POST http://127.0.0.1:8000/documents -F 'file=@新教材.pdf'      # 上传并开始入库 -> 202 {"job_id": "...", "status": "running", ...}
curl -s http://127.0.0.1:8000/ingest/jobs/<job_id>                        # 轮询：pending|running|succeeded|failed
curl -s -X POST http://127.0.0.1:8000/ingest                              # 不上传，只对数据目录重新做一次增量入库
curl -s -X DELETE 'http://127.0.0.1:8000/documents/新教材.pdf'
```

- 任务状态字段：`job_id`、`status`、`files_total`、`files_done`、`current_file`、`chunks_added`、`chunks_deleted`、`skipped_files`（旧版 `.doc` 和入库失败的文件）、`error`、`started_at`、`finished_at`。`succeeded` 表示入库完成**且 BM25 已刷新**；有文件失败时整体 `failed`，其余文件已正常入库并写入元数据。
- `POST /documents` 校验：扩展名（不区分大小写，支持 `.pdf/.docx/.md/.markdown/.txt`，`.doc` 提示另存为 `.docx`）、大小（`RAG_MAX_UPLOAD_MB`，默认 300，流式写盘不整体读入内存，超限 413）、文件名只取最后一段（`../x.pdf` 会存成 `x.pdf`），空 / 隐藏 / `~$` / `README` 开头的名字返回 400；同名文件覆盖（原子替换）。已有任务在运行时文件仍会保存，但返回 409，稍后调用 `POST /ingest` 即可。
- **并发**：入库每 64 个片段为一批（向量化 + 写入）单独持有 `pipeline.lock`，批间释放，所以入库期间 `/search` 可以插进来（会比平时慢一点，本地模型同一时刻只服务一个调用）。同名修改的文件先 upsert 新片段、再删不再存在的旧片段，重建期间该文件仍可检索。任务结束后重建 BM25 快照并原子替换。删除文档期间也算“忙”，此时新任务 / 其他删除返回 409。
- 旧版索引（片段无 `fid`、ID 是 UUID）第一次 `POST /ingest` 或上传时会迁移：按 ID 分批清空旧片段（不删除 collection）再逐文件重建，耗时与全量入库相当。`DELETE` 对旧片段按 `source` 的文件名匹配。

**配置**（环境变量或 `.env`，环境变量优先；[config.py](config.py) 是唯一来源，`query.py` / `evaluate.py` 的默认值也取自这里，命令行参数仍可覆盖）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAG_DEVICE` | `auto` | 本地模型设备：`auto`（mps > cuda > cpu）/ `mps` / `cuda` / `cpu` |
| `RAG_MODE` | `hybrid` | 默认召回方式 |
| `RAG_RETRIEVE_K` | `20` | 精排前的候选数 |
| `RAG_PER_ROUTE_K` | `30` | 混合检索每一路各取多少 |
| `RAG_RERANK_K` | `5` | 精排后保留的片段数 |
| `RAG_MIN_SCORE` | `0.7` | 拒答阈值，0 表示不拒答 |
| `RAG_CONTEXT_MIN_SCORE` | `0.95` | 交给大模型的片段最低精排得分，0 表示不过滤 |
| `RAG_DATA_DIR` / `RAG_DB_DIR` | 仓库内 `data/` / `chroma_db/` | 文档目录 / Chroma 持久化目录（命令行脚本和服务共用） |
| `RAG_META_FILE` / `RAG_CACHE_DIR` | 仓库内 `.ingest_meta.json` / `.bm25_cache/` | 入库元数据文件 / BM25 分词缓存目录 |
| `RAG_MAX_UPLOAD_MB` | `300` | `POST /documents` 单文件大小上限 |
| `RAG_AGENT_MAX_TOOL_CALLS` | `4` | 对话 Agent 每轮用户消息最多调用几次工具 |
| `RAG_AGENT_MAX_MODEL_CALLS` | `6` | 对话 Agent 每轮用户消息最多调用几次大模型（至少 2） |
| `RAG_CHAT_HISTORY_TURNS` | `10` | 每个会话保留最近几轮对话 |
| `RAG_CHAT_MAX_SESSIONS` | `200` | 内存中最多保留的会话数 |
| `RAG_CHAT_SESSION_TTL` | `3600` | 会话空闲多少秒后过期 |
| `RAG_LOG_LEVEL` | `INFO` | 日志级别 |
| `RAG_HOST` / `RAG_PORT` | `127.0.0.1` / `8000` | `uv run api.py` 的监听地址 |

**大模型配置**（`llm_protocol` 目前只实现 `openai`，即 OpenAI 兼容的 `/chat/completions`；新增协议只需在 [llm_factory.py](llm_factory.py) 里加一个分支）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAG_LLM_PROTOCOL` | `openai` | 协议；不支持的值会在启动时报错 |
| `RAG_LLM_BASE_URL` | 无 | 接口地址，如 `https://api.openai.com/v1`（兼容旧名 `ARK_BASE_URL`） |
| `RAG_LLM_API_KEY` | 无 | API Key（以 `SecretStr` 保存，不会出现在日志 / `repr` 中；兼容旧名 `ARK_CHAT_API_KEY`） |
| `RAG_LLM_MODEL` | 无 | 模型名（兼容旧名 `ARK_CHAT_MODEL`，方舟填推理接入点 ID） |
| `RAG_LLM_TEMPERATURE` | `0` | 采样温度 |
| `RAG_LLM_MAX_TOKENS` | 不限 | 单次回答的最大输出 token 数 |
| `RAG_LLM_TIMEOUT` | `60` | 单次请求超时（秒） |
| `RAG_LLM_MAX_RETRIES` | `2` | 超时 / 连接错误 / 429 / 5xx 的自动重试次数（指数退避），所以最坏等待约 `TIMEOUT × (RETRIES + 1)` 加退避间隔 |
| `RAG_EVAL_LLM_MODEL` / `RAG_EVAL_LLM_BASE_URL` / `RAG_EVAL_LLM_API_KEY` | 沿用上面的回答模型 | 评测裁判（`evaluate.py` 的 Ragas 与中文裁判）单独用另一个模型 / 服务；设置了哪项就覆盖哪项，其余沿用 |

示例（任意 OpenAI 兼容服务商）：

```dotenv
# OpenAI
RAG_LLM_BASE_URL=https://api.openai.com/v1
RAG_LLM_API_KEY=sk-...
RAG_LLM_MODEL=gpt-4o-mini
# 本地 Ollama / vLLM 等：RAG_LLM_BASE_URL=http://127.0.0.1:11434/v1，RAG_LLM_API_KEY 随便填一个非空值
# 评测用更强的模型当裁判（可选，只设模型名就沿用上面的地址和密钥）
RAG_EVAL_LLM_MODEL=gpt-4o
```

- 优先级：`RAG_LLM_*` > 旧的 `ARK_*`。注意 pydantic-settings 的来源顺序是“环境变量 > `.env`”，所以**shell 环境变量里的 `ARK_*` 会压过 `.env` 里的 `RAG_LLM_*`**（同一来源内才是 `RAG_LLM_*` 优先）；迁移时请把旧变量改名，不要两套混用。
- 三项（地址、密钥、模型）缺任何一项都算未配置：服务仍可启动，`/search` 可用，`/ask` 返回 503。
- 模型回答超时：底层客户端先按 `RAG_LLM_MAX_RETRIES` 重试，仍超时则 `/ask` 返回 **504**（`llm_timeout`）；其他失败返回 502。完整异常只写日志，响应里不含密钥或上游地址。

- 运行全部单元测试：`uv run python tests/run_all.py`（每个 `tests/test_*.py` 一个子进程，打印各自汇总行，任一失败则非零退出；可带文件名片段只跑部分，如 `run_all.py llm api`）。CI：`.github/workflows/tests.yml`（push / PR 到 main 时在 ubuntu 上用 CPU 版 torch 跑同一个脚本）。
- 单元测试（均不加载模型、不碰真实数据，网络只访问本机 127.0.0.1 的桩服务）：`uv run python tests/test_api.py`（假流水线 / 假入库，另有用临时 Chroma 的端到端用例）、`tests/test_ingest_service.py`（临时目录 + 假 embedding）、`tests/test_sources.py`（按书过滤、快照刷新、分词缓存）、`tests/test_toc.py`（用 fitz 生成带书签的 PDF）、`tests/test_config.py`、`tests/test_llm.py`（本机 OpenAI 兼容桩服务：参数透传、5xx 重试、超时 -> 504、`/ask` 端到端）、`tests/test_parser.py`（临时生成的 PDF / DOCX / GB18030 文本）。
- 依赖：文件上传需要 `python-multipart`（已加入 `requirements.txt` / `requirements.lock`，旧环境请重新执行 `uv pip sync requirements.lock`）。

## 对话 Agent

在 `/ask`（单轮、固定流程）之外，新增多轮对话 Agent（Stage 4）：由大模型决定是否检索、检索什么，再带 `[n]` 引用作答。`/ask` 不受影响。

```text
用户消息 → Agent（create_agent，LangGraph）─┬→ search_knowledge(query, sources?) → 召回 + 精排 + 拒答 / 上下文过滤（同 /ask）
                                           ├→ list_documents()                  → 书目与片段数
                                           └→ 最终回答（[n] 引用）
会话历史：InMemorySaver，thread_id = session_id
```

- 代码：[agent_tools.py](agent_tools.py)（工具与引用登记表）、[chat_agent.py](chat_agent.py)（Agent、会话、限制）、[chat.py](chat.py)（命令行）。
- **多轮改写**不单独做一步：工具描述要求模型把 `query` 写成自包含的完整问题（补全代词 / 省略）；检索直接用它。
- **引用**：一轮用户消息内模型可能多次检索，片段编号全轮唯一、同一 `chunk_id` 编号不变（登记表走 LangGraph 运行时上下文）；响应的 `citations` 只含回答里实际标注了 `[n]` 的片段（`n` 可能不连续，只对本轮有效，下一轮重新编号）。
- **限制**：每轮工具调用上限（超出后工具被拒绝，模型据已有结果作答）、每轮模型调用上限（超出后直接结束并提示）；历史只保留最近 `RAG_CHAT_HISTORY_TURNS` 轮（整轮裁剪，不额外调用大模型做摘要）；同一会话的请求串行；失败 / 断开的那一轮会回滚。
- 大模型未配置时 `agent_ready=false`，对话接口返回 503；大模型超时 / 失败沿用 504 / 502。

| 接口 | 说明 |
| --- | --- |
| `POST /chat` | `{session_id?, message, sources?}` → `{session_id, answer, citations, tool_calls, took_ms}`；`session_id` 不传则新建（字母 / 数字 / `_` / `-`，最长 64）；`message` 1~2000 字；`sources` 作为检索的默认书目过滤（模型显式传了 `sources` 则以模型为准） |
| `POST /chat/stream` | 同上，SSE 流式。事件：`session`、`tool_call`（`name` / `args`）、`token`（`text`）、`citations`、`done`（`answer` / `took_ms`）、`error`（统一错误 JSON） |
| `GET /chat/sessions` | 会话列表 `{sessions:[{session_id, title, updated_at, message_count}]}`，按最近使用倒序；`title` 是第一条用户消息（最多 30 字）；Agent 未就绪时返回空列表（不是 503） |
| `GET /chat/sessions/{id}` | 消息历史（`role` + `content`，工具消息只给摘要）；不存在 / 已清空 / 已过期返回 404 |
| `DELETE /chat/sessions/{id}` | 清空会话 |

```bash
curl -s -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' \
  -d '{"session_id": "demo", "message": "什么是挣值分析？"}'
curl -s -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' \
  -d '{"session_id": "demo", "message": "它的进度绩效指数怎么算？"}'        # 追问：同一个 session_id
curl -N -X POST http://127.0.0.1:8000/chat/stream -H 'Content-Type: application/json' \
  -d '{"session_id": "demo", "message": "再举个例子"}'                       # SSE：-N 关闭 curl 缓冲
curl -s http://127.0.0.1:8000/chat/sessions/demo
curl -s -X DELETE http://127.0.0.1:8000/chat/sessions/demo
```

命令行：`uv run chat.py [--source 书名.pdf]`（`/new` 新会话，`exit` 退出）。

局限：
- 会话只存在服务进程内存里，重启即丢；超过 `RAG_CHAT_MAX_SESSIONS` 或空闲超过 `RAG_CHAT_SESSION_TTL` 的会话被淘汰（不持久化、不跨进程）。
- **尚未用真实大模型验证**：单元测试用脚本化的假模型；豆包 / 其他 OpenAI 兼容服务需支持 `tools` / `tool_calls`，这是假设，未验证。
- 流式 `token` 是模型每次生成的文本增量；若模型在调用工具前先说了一句话，这句话也会出现在 token 流里，以 `done.answer` 为准。
- 单元测试：`uv run python tests/test_agent.py`（假检索 + 假模型，不调用真实大模型）。

## Web 界面

`web/` 是前端工程（React + Vite + TypeScript + Tailwind CSS + shadcn/ui；路由 react-router，数据请求 TanStack Query，回答用 react-markdown 渲染），Stage 5 新增。需要 Node 22 与 pnpm。

```bash
# 开发：后端在 8000 端口运行（uv run api.py），前端 dev server 把 API 请求代理过去
cd web && pnpm install && pnpm dev          # http://localhost:5173，热更新

# 生产：构建后由 FastAPI 在同一端口托管，不需要另起前端服务
cd web && pnpm install && pnpm build        # 产物在 web/dist
uv run api.py                               # 浏览器打开 http://127.0.0.1:8000/
```

- **托管方式**：`web/dist` 存在时，API 服务提供 `/assets/*`（带内容哈希，长缓存）和前端路由 `/`、`/library`、`/c/{session_id}`（返回 `index.html`）；这些路径不会遮住 `GET /documents`、`/chat/sessions`、`/health` 等 JSON 接口。目录是否存在在请求时检查，服务运行中重新 `pnpm build` 立即生效；未构建时 `GET /` 返回提示“前端未构建：cd web && pnpm install && pnpm build”。构建目录可用 `RAG_WEB_DIST` 改（相对路径按仓库根目录解析）。
- **对话页**（`/`、`/c/:id`）：左侧会话列表（`GET /chat/sessions`，可删除）；中间流式回答（`POST /chat/stream`，可随时“停止”）；回答里的 `[n]` 变成可悬停预览的引用小标，点击在右侧“依据”面板定位原文（书名、章节、页码、片段、精排得分）；检索步骤显示为回答上方的淡色折叠行；输入框旁的“范围”可限定只在某几本书里检索。窄屏（<1024px）侧栏、（<1280px）依据面板改为抽屉。
- **知识库页**（`/library`）：书目卡片（片段数、章节覆盖率、删除）、拖拽或点击上传（`.pdf` `.docx` `.md` `.markdown` `.txt`）、“重新扫描”，以及入库任务的实时进度（轮询 `GET /ingest/jobs/{id}`，刷新页面后仍会继续显示）。
- **状态指示**（右上角）：绿色“就绪”（索引与 Agent 都可用）、琥珀色“仅检索”（大模型未配置）、红色“未就绪 / 离线”；点开看片段数、BM25 构建时间、设备、召回模式、大模型和 Agent 状态。每 15 秒轮询 `/health`。
- 书名显示去掉了扩展名和 z-library 之类的下载站后缀（`web/src/lib/books.ts`），完整文件名在悬停提示里。
- 主题：浅色 / 深色 / 跟随系统，保存在浏览器 localStorage。
- 检查：`cd web && pnpm build`（TypeScript 严格类型检查 + 打包）、`pnpm lint`（oxlint）。后端侧的单元测试：`uv run python tests/test_web.py`（会话列表、静态托管与 SPA 回退）。

局限：
- 历史会话重新打开时，后端只保存消息文本，不保存当时的引用片段，所以旧回答里的 `[n]` 只显示为普通文字（本页面会话内切换时引用保留在内存里，刷新页面后丢失）。
- 会话只存在服务进程内存里（见“对话 Agent”），重启服务后会话列表为空。
- 上传一次一个文件（后端同一时刻只有一个入库任务）。
- 没有搜索实验页；`/search` 接口仍可直接调用。

## 性能设计

- 已下载的 Qwen 模型优先从本机缓存加载，缓存缺失时才访问 Hugging Face Hub。
- Embedding 与 Reranker 在 MPS / CUDA 上使用 FP16（CPU 保持默认精度）。
- Reranker 使用小批量推理并在启动时预热 MPS kernel。
- 查询 embedding 使用有上限的进程内 LRU 缓存，精排分数使用有上限的进程内字典缓存。
- Ragas 五项指标与两个中文裁判并发执行，并复用同一事件循环和连接池。

## 关键参数

切分参数直接定义在代码中，其余集中在配置里：

- 切分大小与重叠：[rag_common.py](rag_common.py)
- 召回数、精排数、阈值、设备、服务地址、前端构建目录：[config.py](config.py)（均可用 `RAG_*` 环境变量覆盖，见“服务化运行”）；`query.py` / `evaluate.py` 的命令行参数优先级更高
- 本地模型与设备：[embed.py](embed.py)、[reranker.py](reranker.py)

扫描版 PDF 或文本层质量较差的 PDF 需要另行增加 OCR；当前项目不会自动 OCR。

## 常见问题

- `AuthenticationError`：检查 `.env` 中的 `RAG_LLM_API_KEY`、`RAG_LLM_MODEL` 和 `RAG_LLM_BASE_URL`（旧的 `ARK_*` 同样有效）。
- 找不到向量库：先运行 `uv run ingest.py`。
- MPS 不可用：设置环境变量 `RAG_DEVICE=cpu`（默认 `auto` 会自动回落到 CUDA / CPU）。
- 依赖环境异常：重新执行 `uv pip sync requirements.lock`。
