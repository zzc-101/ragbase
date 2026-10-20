# 体检（ragbase diagnose）

`ragbase diagnose` 给你的知识库出一份“体检报告”：用可量化的数据看清**当前参数**、**各阶段的召回质量**和**各阶段的耗时**，再附上几条**建议**。

它只测量和建议：

- 不改任何配置，不自动扫参数——要不要调、调哪个，由你决定；
- **不调用大模型**，没有大模型费用（本地模型照常运行；云端 embedding / 精排接口照常调用，会产生它们各自的少量费用）；
- 评测（[`ragbase eval`](evaluation.md)）管**回答与检索的质量**，体检管**参数是否合理、哪一步慢**。两者复用同一份题集 `eval_questions.json`。

## 运行

```bash
uv run ragbase diagnose                          # 用全部题，召回方式取 RAG_MODE（默认 hybrid）
uv run ragbase diagnose --limit 10               # 只用前 10 道题，先冒烟
uv run ragbase diagnose --retrieval-mode vector  # 指定召回方式：vector / bm25 / hybrid
```

前置条件与 `ragbase eval` 一样：已经 `ragbase ingest` 建好索引，项目主目录（`RAG_HOME`）下有 `eval_questions.json`（格式见[评测题格式](evaluation.md#评测题格式)）。索引为空、题集缺失、embedding 配置与索引不一致、云端提供方配置有误时，都会给出中文说明后退出。

运行过程：读题集（同文本重复的问题只测一次）→ 构建流水线并记录加载耗时 → 用一个固定的假问题预热一遍 → 逐题做“召回 + 精排”并分阶段计时 → 打印报告 → 把完整数据写入 `RAG_HOME/diagnose_report.json`（已加入 `.gitignore`）。

## 报告怎么看

报告分四块：当前参数、质量、速度、建议。

### 当前参数

| 项 | 含义 |
| --- | --- |
| 召回方式 | `vector` / `bm25` / `hybrid` |
| `per_route_k` | 混合召回每一路各取多少（`RAG_PER_ROUTE_K`；只在 hybrid 模式生效） |
| `retrieve_k` | 交给精排的候选数（`RAG_RETRIEVE_K`）；单路模式下也是该路的召回数 |
| `rerank_k` | 精排后保留的片段数（`RAG_RERANK_K`） |
| `min_score` / `context_min_score` | 拒答阈值 / 交给大模型的片段的得分下限（`RAG_MIN_SCORE` / `RAG_CONTEXT_MIN_SCORE`） |
| embedding、reranker | 提供方与模型名 |
| reranker 批大小 | 本地精排每批送进模型的候选数；云端 / 不精排时显示“不适用” |
| 计算设备 | 本地模型用的设备（`mps` / `cuda` / `cpu`）；全云端为 `cloud` |
| 片段数、片段长度 | 索引里的片段总数，以及正文字符数的均值 / 中位数 / 最大值 |
| 切分默认值 | 代码里的 `chunk_size` / `overlap` 默认值（`ingest/splitter.py`），不是对索引的实测——你的索引可能是别的参数切的 |
| 模型加载耗时 | 构建流水线的整体耗时（模型加载 + BM25 索引快照构建） |
| 参与统计的题数 | 实际测量的题数；其中有参考答案的另列。没有参考答案的题只计耗时，不计召回率 |

### 质量

沿用 `ragbase eval` 的**词面代理指标**（算法见[词面指标怎么算](evaluation.md#词面指标怎么算)），分三组：

| 组 | 含义 |
| --- | --- |
| 召回 TopK 候选 | 进入精排之前的候选（`retrieve_k` 条） |
| 精排后 | 精排保留的 `rerank_k` 条 |
| 交给大模型的片段 | 精排后再经拒答和上下文过滤的片段（口径与 `eval --retrieval-only` 完全一致；被拒答的题这一组为空） |

每组给出 `point_recall`、`context_precision`、`mrr` 的均值。此外：

- **拒答率**：最相关片段得分低于 `min_score`（或召回为空）的题占比；
- **Top1 精排得分分布**：最小 / 中位数 / 最大。reranker 为 `none` 时没有精排得分；
- **平均交给大模型的片段数**。

### 速度

每题一次“召回 + 精排”，按阶段计时（毫秒），给出均值 / 中位数 / 最大值，以及各阶段占“检索总耗时”的百分比：

| 阶段 | 计时范围 |
| --- | --- |
| 向量化查询（embed） | `embeddings.embed_query` |
| 向量库查询 | `store.query` |
| BM25 关键词 | BM25 整路，含 jieba 分词 |
| RRF 融合 + 组装 | 融合各路排名并组装 Document |
| 精排（rerank） | `reranker.rerank_with_scores` |

- 没跑的阶段不显示（例如 `vector` 模式没有 BM25）。
- 精排另给“每条候选平均毫秒”= 精排总耗时 ÷ 候选总条数，用来估算调整 `RAG_RETRIEVE_K` 的收益。
- embedding 或精排用云端时，对应行标注“含网络往返”：耗时里包含网络延迟，不只是服务端计算。
- 占比相加可能略低于 100%，差额是阶段之间的少量调度开销。

### 建议

按下面的固定规则触发，每条分三行：**现象**（带具体数字）/ **可以尝试**（写明环境变量）/ **可能的代价**。没有规则触发时只打印“未发现明显问题”。拿不准的情况不给建议。

质量类规则（c / d / e / f）需要至少 5 道题的样本才会触发（c / d 看有参考答案的题数，e 看没被拒答的有参考答案的题数，f 看参与统计的题数）。

| 规则 | 触发条件（初始阈值） | 建议方向 |
| --- | --- | --- |
| a. 精排拖慢检索 | 精排占检索总耗时 ≥ 60% | 减少 `RAG_RETRIEVE_K`；本地精排时还可换更小的精排模型（`RAG_RERANKER_MODEL`，需重新校准 `RAG_MIN_SCORE`）、改云端精排（`RAG_RERANKER_PROVIDER=api`）；设备为 `cpu` 时再加一句用 GPU |
| b. 本地模型跑在 CPU | 有本地模型且设备为 `cpu` | 有 GPU / Apple 芯片时设 `RAG_DEVICE`；或改用云端模型 |
| c. 召回阶段漏掉答案 | 召回 TopK 的 `point_recall` 均值 < 0.6 | 改用 hybrid（若当前不是）、增大 `RAG_PER_ROUTE_K` / `RAG_RETRIEVE_K`、检查切分和 `bm25_userdict.txt` |
| d. 精排把相关片段排出保留范围 | 召回 TopK 比精排后的 `point_recall` 高出 > 0.15 | 增大 `RAG_RERANK_K`；或换更准的精排模型 |
| e. 上下文过滤偏严 | 精排后比交给大模型的片段的 `point_recall` 高出 > 0.10 | 降低 `RAG_CONTEXT_MIN_SCORE` |
| f. 拒答率偏高 | 拒答率 > 20%（reranker 为 `none` 时不适用） | 若题目的答案都应在库里，`RAG_MIN_SCORE` 可能偏高，用 `scripts/calibrate_rerank_threshold.py` 重新校准 |
| g. 没有精排 | reranker 为 `none` | 提示没有精排就不会按相关度拒答 |
| h. 云端调用延迟偏大 | 云端 embedding 或精排单次均值 > 1000 ms | 换就近的云端服务，或改用本地模型 |

说明：

- 规则 e 只比较**没被拒答**的题：被拒答的题没有片段交给大模型，它造成的“损失”属于拒答（规则 f），不算在上下文过滤头上。报告“质量”表里的“交给大模型的片段”一行仍按 eval 的口径把拒答题算作空，所以它可能低于规则 e 引用的数字。
- 阈值是模块 `ragbase/diagnostics.py` 顶部的常量（`RERANK_SHARE_PCT`、`CANDIDATE_RECALL_LOW`、`RERANK_RECALL_DROP`、`CONTEXT_RECALL_DROP`、`REFUSAL_RATE_HIGH`、`CLOUD_LATENCY_MS`、`MIN_QUESTIONS_FOR_QUALITY_RULES`），是初始值，可按经验调整。

## 输出文件

完整数据写入 `RAG_HOME/diagnose_report.json`（UTF-8，原子写入），结构：

| 键 | 内容 |
| --- | --- |
| `generated_at` | 生成时间 |
| `params` | “当前参数”那一块的全部数据 |
| `warmup_question` | 预热用的假问题（不计入统计） |
| `questions` | 逐题记录：各阶段耗时（`timings_ms`，含 `total`）、候选 / 精排 / 交给大模型的片段数、Top1 得分、是否拒答、三组词面指标（无参考答案的题为 `null`） |
| `summary` | 汇总：题数、质量、速度 |
| `recommendations` | 触发的建议（`id` / `title` / `phenomenon` / `try` / `cost`） |

## 注意事项

- **只有相对意义**：词面指标是基于参考答案要点的代理指标，不是人工标注的标准 Precision / Recall；题集越贴近你的真实提问，结果越有参考价值。调整参数后重新运行 `ragbase diagnose`，对比前后变化，而不是和别处的绝对数字比。
- **云端耗时含网络**：云端 embedding / 精排的耗时包括网络往返，会随时间和网络状况波动；多跑几次看趋势。
- **预热**：正式计时前会用一个不在题集里的固定假问题跑一遍完整的召回 + 精排，不计入统计，避免把 MPS kernel 编译、首次调用开销算进第一道题。
- **重复问题去重**：embedding 和精排都有进程内缓存，同一个问题第二次几乎是 0 毫秒，会让耗时失真，所以同文本重复的问题只测一次（报告里会注明跳过了几道）。
- **`--limit`** 取题集前 N 道（再去重）。样本太少时，质量类建议不会触发。
- 本命令只做读取和计时，不会修改索引或配置。用嵌入式 Qdrant 时，同一目录同一时刻只允许一个进程打开，所以不要和运行中的 `ragbase serve` 同时运行（见[部署](deployment.md)）。
