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
- `rag_common.py`：稳定 ID、文件指纹和入库编排

## 混合检索

[hybrid_retriever.py](hybrid_retriever.py) 同时走两路召回：向量路直接查 Chroma（各取 30），BM25 路用 `jieba` 分词 + `bm25s` 打分（各取 30），再用 RRF（`score = Σ 1/(60+rank)`，只看排名）融合取前 20 交给 Reranker。专有名词、缩写和公式（如 TOGAF、SPI）这类向量容易漏的内容，由 BM25 补召回。

- BM25 索引在启动时从 Chroma 已有的片段文本现建，无需改动入库流程。
- 分词结果缓存在 `.bm25_cache/`（已忽略），片段数、`.ingest_meta.json` 或用户词典变化后自动重建。
- 领域词典放在仓库根目录的 [bm25_userdict.txt](bm25_userdict.txt)（jieba 格式：`词 词频 词性`）。不要放进 `data/`，那里的 `.txt` 会被当作文档入库。修改后下次启动自动生效。
- 召回模式用 `--mode` 切换：`uv run query.py --mode {vector,bm25,hybrid}`（默认 `hybrid`）。答案后的来源会标出召回路径，如 `[V3+B1]` 表示向量路第 3、BM25 路第 1。
- 单元测试：`uv run python tests/test_hybrid.py`。

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
ARK_CHAT_API_KEY=你的方舟 API Key
ARK_CHAT_MODEL=你的推理接入点 ID
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
CHROMA_TELEMETRY_DISABLED=true
```

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

全量入库和增量入库使用相同的稳定片段 ID，并通过 `.ingest_meta.json` 记录文件状态。旧版索引首次执行增量命令时会自动迁移重建。

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
- 启动约 10 秒（加载两个本地模型和 BM25 索引）。索引不存在或为空时服务仍会启动，`/health` 报告 `not_ready`，`/search` 与 `/ask` 返回 503（“索引不存在或为空，请先运行 uv run ingest.py”）；入库后需重启服务。
- 每个响应带 `X-Request-ID`（请求头里带了就沿用，否则生成），每个请求输出一行访问日志（方法、路径、状态码、耗时；`/search`、`/ask` 另带 `refused` 和 `top_score`）。
- 错误统一为 `{"error": "...", "detail": "..."}`：422 参数校验、400 请求了未构建的 BM25 模式、502 大模型调用失败、503 索引 / 大模型未就绪、500 未处理异常（详情只写日志）。
- 交互式文档：<http://127.0.0.1:8000/docs>。

| 接口 | 说明 |
| --- | --- |
| `GET /health` | `status`（`ok` / `not_ready`）、`chunk_count`、`bm25_ready`、`device`、`mode`、`llm_configured` |
| `POST /search` | 召回 + 精排，不调用大模型 |
| `POST /ask` | 召回 + 精排 + 大模型回答；得分过低时拒答且不调用大模型 |

`/search` 请求：`query`（1~1000 字，首尾空白会被去掉）必填；可选 `mode`（`vector` / `bm25` / `hybrid`）、`top_k`（精排后返回数，默认 5）、`min_score`（0 表示不拒答）、`include_candidates`（同时返回精排前的候选，默认 false）。

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
     "scores": {"vec_rank": 2, "bm25_rank": 2, "rrf_score": 0.0323, "rerank_score": 0.9996}}
  ],
  "took_ms": 3512.4
}
```

`refused` 为 true 表示最相关片段得分低于 `min_score`（`chunks` 仍会返回，便于查看）；`page` 为 1 起的页码，仅 PDF 有，否则为 null。

`/ask` 请求：`query` 必填；可选 `mode`、`min_score`、`context_min_score`。

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

`citations[n-1]` 对应回答中的 `[片段n]`；拒答时 `citations` 为空。

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
| `RAG_LOG_LEVEL` | `INFO` | 日志级别 |
| `RAG_HOST` / `RAG_PORT` | `127.0.0.1` / `8000` | `uv run api.py` 的监听地址 |
| `ARK_CHAT_API_KEY` | 无 | 方舟 API Key（以 `SecretStr` 保存，不会出现在日志 / `repr` 中） |
| `ARK_CHAT_MODEL` | 无 | 推理接入点 ID |
| `ARK_BASE_URL` | 无 | 方舟地址，如 `https://ark.cn-beijing.volces.com/api/v3` |

未配置 `ARK_*` 时服务仍可启动，`/search` 可用，`/ask` 返回 503。

- 单元测试：`uv run python tests/test_api.py`（假流水线，不加载模型、不调用大模型）、`tests/test_config.py`。

## 性能设计

- 已下载的 Qwen 模型优先从本机缓存加载，缓存缺失时才访问 Hugging Face Hub。
- Embedding 与 Reranker 在 MPS / CUDA 上使用 FP16（CPU 保持默认精度）。
- Reranker 使用小批量推理并在启动时预热 MPS kernel。
- 查询 embedding 使用有上限的进程内 LRU 缓存，精排分数使用有上限的进程内字典缓存。
- Ragas 五项指标与两个中文裁判并发执行，并复用同一事件循环和连接池。

## 关键参数

切分参数直接定义在代码中，其余集中在配置里：

- 切分大小与重叠：[rag_common.py](rag_common.py)
- 召回数、精排数、阈值、设备、服务地址：[config.py](config.py)（均可用 `RAG_*` 环境变量覆盖，见“服务化运行”）；`query.py` / `evaluate.py` 的命令行参数优先级更高
- 本地模型与设备：[embed.py](embed.py)、[reranker.py](reranker.py)

扫描版 PDF 或文本层质量较差的 PDF 需要另行增加 OCR；当前项目不会自动 OCR。

## 常见问题

- `AuthenticationError`：检查 `.env` 中的方舟 API Key、模型接入点和地址。
- 找不到向量库：先运行 `uv run ingest.py`。
- MPS 不可用：设置环境变量 `RAG_DEVICE=cpu`（默认 `auto` 会自动回落到 CUDA / CPU）。
- 依赖环境异常：重新执行 `uv pip sync requirements.lock`。
