# 评测

> 评测（`ragbase eval`）管**回答与检索的质量**；想看**当前参数是否合理、各阶段有多慢、该调哪个环境变量**，用 [`ragbase diagnose`](diagnostics.md)（同样复用本题集，不调用大模型）。

`ragbase eval` 对一组带参考要点的问题跑完整的“召回 → 精排 → 拒答 → 生成”流程，并给出三层指标。评测也是调拒答阈值、对比召回方式、验证换模型有没有变好的主要工具。

> 仓库根目录自带的 `eval_questions.json`（30 题）是针对作者自己的语料（信息系统项目管理师 / 系统架构设计师教材）写的，首次打开数据库时会被自动导入题集。用在你自己的资料上，请先换成你的题目，见[评测题格式](#评测题格式)。

## 三层指标

1. **Ragas**（需要大模型，会产生费用）：`faithfulness`（忠实度）、`answer_relevancy`（回答相关性）、`factual_correctness`（事实正确性）、`context_precision`（上下文精确率，带参考答案）、`context_recall`（上下文召回率）。用评测裁判模型（`RAG_EVAL_LLM_*`，缺省沿用回答模型）；`answer_relevancy` 还会用项目已加载的 embedding。调用缓存在 `RAG_HOME/.ragas_cache/`。
2. **词面代理指标**（不需要大模型，免费）：`point_recall`、`context_precision`、`mrr`，分别统计三组片段——召回 Top20（`retrieved_top20`）、精排 Top5（`reranked_top5`）、实际交给大模型的片段（`llm_context`）。
3. **原有中文裁判**（需要大模型）：回答裁判（`正确` / `部分正确` / `错误`）与检索裁判（`充分` / `部分` / `不足`，看的是召回 Top20 是否足以回答）。

### 词面指标怎么算

没有人工标注的“相关文档”，所以用参考要点做代理：

1. 把参考答案按标点（`，,；;。！？!?、：:`）拆成若干**要点**，每个要点规范化（小写，只保留数字、字母、汉字），丢弃不足 2 个字符的。
2. 一个要点“命中”某个片段：要点是片段的子串，或要点的字符二元组（bigram）有 ≥ 35% 出现在片段里。
3. `point_recall` = 被至少一个片段命中的要点占比；`context_precision` = 命中了至少一个要点的片段占比；`mrr` = 第一个命中要点的片段排名的倒数（都没命中为 0）。没有片段或没有要点时三者都为 0。

> 词面指标是基于参考答案要点的代理指标，不是基于人工相关文档标注的标准 Precision / Recall / MRR——用来在同一份题集上做**相对**比较（比如换召回方式、换 reranker 之前 / 之后），不要和别处的绝对数字比。

## 运行

```bash
uv run ragbase eval                        # 全部题：生成回答 + Ragas + 中文裁判（会调用大模型，产生费用）
uv run ragbase eval --limit 1              # 先用一道题冒烟测试
uv run ragbase eval --skip-ragas --skip-legacy-judge   # 只做词面指标和问答生成，跳过评测模型的额外调用（回答仍会调用大模型）
uv run ragbase eval --retrieval-only       # 只跑召回 + 精排和词面指标：不调用任何大模型，没有 API 费用
```

| 参数 | 说明 |
| --- | --- |
| `--limit N` | 只评估前 N 道题（在 `--start-id` 过滤之后取） |
| `--start-id N` | 从题号 `id ≥ N` 开始，默认 1（`id` 是题集里的题号，见[评测题格式](#评测题格式)） |
| `--skip-ragas` | 跳过 Ragas 指标 |
| `--skip-legacy-judge` | 跳过中文 LLM 裁判 |
| `--retrieval-mode {vector,bm25,hybrid}` | 召回方式，默认取 `RAG_MODE`（`hybrid`） |
| `--retrieval-only` | 只评估召回与精排的词面指标，不调用 LLM / Ragas / 裁判（也不需要配置大模型） |
| `--min-score X` | 拒答阈值，默认 `RAG_MIN_SCORE`；`0` 不拒答 |
| `--context-min-score X` | 交给大模型的片段下限，默认 `RAG_CONTEXT_MIN_SCORE`；`0` 不过滤 |

说明：

- 评测要求索引已建好；embedding 配置与索引不一致时会给出说明后退出（见 [models.md](models.md#换-embedding索引签名规则)）。
- 题目从内置数据库读取（见[评测题格式](#评测题格式)）；题集为空时会提示先导入或在 Web 上添加。**`reference` 为空的题没有参考答案，评测会跳过它们**并打印被跳过的题号（`ragbase diagnose` 则照常计耗时、只是不计召回率）。
- 评测只测单轮问答流程（与 `/ask` 同一套），**不涉及对话 Agent**，也没有按书过滤参数。
- 完整评测会调用回答模型（每题一次）、评测裁判模型（Ragas 五项指标 + 两个裁判，并发执行）；具体调用次数取决于 Ragas 内部实现，这里不给数字。可以单独用 `RAG_EVAL_LLM_MODEL` 指定更强的裁判模型。
- 因得分过低被拒答的题不会调用回答模型（评测裁判仍会评这条拒答回答）：Ragas 的上下文退回精排 Top5（保持原有行为），其余字段照常记录。

### 对比召回方式

```bash
uv run ragbase eval --retrieval-only --retrieval-mode vector
uv run ragbase eval --retrieval-only --retrieval-mode bm25
uv run ragbase eval --retrieval-only --retrieval-mode hybrid
```

每种模式写入各自的结果文件，互不覆盖。

## 结果

结果写入项目主目录（`RAG_HOME`）下：

| 文件 | 来源 |
| --- | --- |
| `eval_results.<mode>.json` | 完整评测，如 `eval_results.hybrid.json` |
| `eval_results.<mode>.retrieval.json` | `--retrieval-only` |

它们已被 `.gitignore` 排除。每完成一题就原子保存一次，长评测中断后仍保留已有结果。每题一条记录：

| 字段 | 说明 |
| --- | --- |
| `id` / `question` / `reference` / `difficulty` / `type` | 来自题集 |
| `retrieval_mode` | 本次召回方式 |
| `answer` / `evidence` | 回答（已去掉【依据】）与逐字摘录的依据；`--retrieval-only` 时 `answer` 为 `null` |
| `top1_rerank_score` / `refused` | 最相关片段的精排得分、是否会被拒答 |
| `lexical` | `{retrieved_top20, reranked_top5, llm_context}`，每个都是 `{point_recall, context_precision, mrr}` |
| `ragas` | `{scores: {…五项…}, errors: {…}}`；某项失败时该项为 `null` 并记录错误 |
| `legacy_judge` | `{answer, retrieval}` |
| `retrieved_contexts` | 精排 Top5 的片段记录（未过滤）；`llm_contexts` 是大模型实际看到的片段 |

运行结束会打印汇总：Ragas 平均分、**因精排得分过低拒答的题数**、三组词面指标的平均值、交给 LLM 的片段平均个数、两个裁判的标签分布。

## 评测题格式

题集存在内置数据库（`RAG_HOME/ragbase.db`，见[架构：数据库](architecture.md#数据库)）里，`ragbase eval` 和 [`ragbase diagnose`](diagnostics.md) 共用，是唯一来源：

- **首次启动自动导入**：数据库还没做过首次导入、题集为空，并且 `RAG_HOME/eval_questions.json` 存在时，第一次打开数据库就把它导入一次，之后不再导入（详见[体检：题集](diagnostics.md#题集)）。`ragbase init` 不会生成这个文件。
- **之后用命令行或 Web 管理**：`ragbase questions import FILE [--replace]` / `ragbase questions export FILE`，或在 Web 的“体检”页增删改、导入、导出。改仓库根目录的 `eval_questions.json` 不会再影响已经导入过的数据库。

JSON 文件是导入 / 导出用的格式：一个数组，每题一个对象。

```json
[
  {
    "id": 1,
    "question": "什么是软件架构风格？",
    "reference": "描述某一特定应用领域中系统组织方式的惯用模式，定义系统家族、词汇表和约束",
    "source_topic": "架构风格定义",
    "difficulty": "简单",
    "type": "概念题"
  }
]
```

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `question` | 是 | 问题，非空字符串 |
| `reference` | 否 | 参考答案（或参考要点），用于 Ragas 与词面指标。**为空的题 `ragbase eval` 会跳过**；体检照常计耗时，只是不计召回率 |
| `answer_keyword` | 否 | 旧版字段，等同 `reference`（两个都写时取 `reference`）；老的题集文件不用改就能导入。库里不单独存它 |
| `id` | 否 | 正整数，用于 `--start-id`。不写则自动分配；导入时 `id` 重复（文件内）算错误 |
| `difficulty`、`type` | 否 | 自定义标签，只写进结果文件 |
| `source_topic` | 否 | 备注，评测流程不使用 |

导入规则：

- **校验是整体的**：文件不是数组、某题缺 `question`、字段类型不对（如 `id` 不是整数、`reference` 不是字符串）、文件内 `id` 重复，都会让整个文件不导入，并逐题列出问题（命令行逐条打印；HTTP 返回 422 与 `errors`）。
- **追加**（默认）：与库里已有题目 `id` 冲突的题改用新 `id`，并告诉你换了哪些；**替换全部**（`--replace`）：先清空题集，保留文件里的 `id`。
- **导出**：每题同时写出 `reference` 和 `answer_keyword`（取值相同），所以导出的文件旧版 `ragbase eval` 也能直接读。

### 如何新增题目

- 问题按真实用户的口吻写；覆盖不同类型（概念、对比、多方面、公式 / 步骤）。
- `reference` 用**逗号 / 分号分隔的若干要点**，措辞尽量贴近资料原文——词面指标是靠字符二元组匹配的，要点太笼统或用了资料里没有的同义词会算作未命中。每个要点至少 2 个字符。
- 想让 Ragas 的“事实正确性”等指标更可信，把 `reference` 写成完整的参考答案（同时仍包含若干贴近原文的要点，词面指标靠它们匹配）。
- 再准备几道**库外问题**（资料里没有答案的），用来检验拒答；它们放在校准脚本里，不放在这个题集里，见下。

## 校准工作流

调拒答阈值（换了教材、reranker 或召回方式之后）：

1. `uv run python scripts/calibrate_rerank_threshold.py`：不调用大模型，打印库内题与库外问题的最高得分分布、各候选阈值的误拒 / 漏拦数和推荐值。输出的含义见 [models.md](models.md#校准脚本)。脚本里的库外问题和补充题是针对作者语料写的，**换了语料要同步改它们**。
2. 把推荐值写进 `.env` 的 `RAG_MIN_SCORE`（必要时同时调整 `RAG_CONTEXT_MIN_SCORE`）。
3. 用 `uv run ragbase eval --retrieval-only` 验证：汇总里“因精排得分过低拒答”对库内题集应当接近 0；调高阈值能拦下更多库外问题，代价是更容易误拒正常问题。临时试值用 `--min-score`，不用改配置：`uv run ragbase eval --retrieval-only --min-score 0.3`。
4. 需要时再跑完整评测（含生成）确认端到端质量。

`--context-min-score` 同理：`uv run ragbase eval --retrieval-only --context-min-score 0.99` 可以看到调高后 `llm_context` 的 `point_recall` 变化。本地 reranker 的得分很饱和，调高会开始裁掉对比 / 多方面题需要的片段，别盲目调高。
