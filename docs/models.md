# 模型：本地 / 云端、示例组合与换模型的规则

ragbase 用到三类模型，各自独立配置：

| 角色 | 作用 | 可选提供方 | 配置项 |
| --- | --- | --- | --- |
| **embedding** | 把片段和问题变成向量（入库和向量召回） | `local` / `openai` | `RAG_EMBEDDING_*` |
| **reranker** | 对召回的候选逐个打相关度分，决定顺序和是否拒答 | `local` / `api` / `none` | `RAG_RERANKER_*` |
| **大模型** | 生成回答、驱动对话 Agent | OpenAI 兼容协议 | `RAG_LLM_*` |

每一项的变量、默认值见 [configuration.md](configuration.md)，这里讲取舍和规则。

## 本地 vs 云端

| | `local`（默认） | `openai` / `api` | `none` |
| --- | --- | --- | --- |
| **embedding**（`RAG_EMBEDDING_PROVIDER`） | sentence-transformers 加载 Qwen3-Embedding，查询自动加 query prompt | `openai`：OpenAI 兼容的 `POST {地址}/embeddings` | 不支持 |
| **reranker**（`RAG_RERANKER_PROVIDER`） | CrossEncoder 加载 Qwen3-Reranker，得分 = sigmoid(logit) = P(相关) | `api`：通用 `POST {地址}/rerank`，分数原样透传 | 跳过精排：保持召回（RRF）顺序取前 `RAG_RERANK_K` 个；没有得分，**不会按相关度拒答**（只有召回为空才拒答），启动时打印警告 |
| 模型名 | `RAG_*_MODEL` = Hugging Face 仓库名 | `RAG_*_MODEL` = 服务商的模型 ID | — |
| 需要下载 | 是（`ragbase models download`，只下载 `local` 的那个） | 否 | 否 |

要点：

- 没有 GPU 的机器用云端更快：不用下载模型；**embedding 与 reranker 都用云端（或 reranker 为 `none`）时**，`ragbase serve / ask / chat / ingest` 启动不再加载 torch（命令行入口会阻止 langchain 顺带导入 transformers / torch）。用 `uvicorn ragbase.api.app:app` 等其他方式启动时 langchain 仍可能加载它，只影响启动速度和内存，不影响功能。`eval` / `doctor` / `models` 不做这个处理。
- 默认本地模型是 `Qwen/Qwen3-Embedding-0.6B` 与 `Qwen/Qwen3-Reranker-0.6B`，缓存在 Hugging Face 缓存目录里各约 1.1 GB（在开发机上用 `du` 实测）。
- 本地实现是按 Qwen3 系列的用法编写的（embedding 的 query prompt、reranker 的 yes/no logit 打分），**换成别家本地模型不保证可用**；云端模型不受此限制。
- 云端接口的连接参数（地址 / 密钥 / 超时 / 重试）的回退规则见 [configuration.md](configuration.md#云端模型接口的回退规则)：留空时回退到大模型的配置，但密钥只在“同一个服务”时才沿用。

### 云端 embedding 的行为

- 请求：`POST {地址}/embeddings`，体为 `{model, input: [文本…], dimensions?}`；文本按 `RAG_EMBEDDING_BATCH_SIZE` 分批。响应需符合 OpenAI 规范（`data[].embedding` 为浮点数组，不支持 base64 编码返回）。
- 返回的向量会做 L2 归一化，与本地模型的排序保持一致；各向量长度必须一致，设置了 `RAG_EMBEDDING_DIMENSIONS` 却返回别的长度会报错（该服务可能不支持 `dimensions`）。
- 查询向量在进程内有 LRU 缓存（256 条），重复问题不重复调用接口。`RAG_EMBEDDING_QUERY_INSTRUCTION` 只加在查询前面。

### 云端 rerank 的行为

- 请求：`POST {地址}/rerank`，体为 `{model, query, documents, top_n}`；响应取 `results[].index` 与 `results[].relevance_score`（也接受 `score`）。这是 Jina、SiliconFlow、Cohere v2 兼容网关、Xinference、vLLM 等常见的形状，具体某个服务商是否完全兼容需要自己验证。
- 分数**原样透传**，不做归一化；服务端忽略 `top_n` 时 ragbase 自己按分数排序并截断。
- 进程内缓存 128 组（问题 + 候选集 + top_k）的结果。

## 示例组合

> 下面的地址和模型名**只是示例，没有逐一验证**，请以服务商文档为准。完整的 `.env` 片段见 `.env.example` 第 3 节。

**1. 全本地（默认）**：什么都不用配。需要 `ragbase models download`；有 Apple Silicon 或 NVIDIA GPU 时最合适。

```dotenv
# RAG_EMBEDDING_PROVIDER=local
# RAG_RERANKER_PROVIDER=local
```

**2. 全云端**（无 GPU 的机器；SiliconFlow 风格，embedding 和 rerank 用同一家）：

```dotenv
RAG_EMBEDDING_PROVIDER=openai
RAG_EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
RAG_EMBEDDING_API_KEY=请替换为你的密钥
RAG_EMBEDDING_MODEL=BAAI/bge-m3
RAG_RERANKER_PROVIDER=api
RAG_RERANKER_BASE_URL=https://api.siliconflow.cn/v1
RAG_RERANKER_API_KEY=请替换为你的密钥
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

配好后要做两件事：`ragbase ingest --full` 重建索引（见下），并按[阈值重新标定](#reranker-换模型后重新标定拒答阈值)。

**3. 混合**：云端 embedding（入库和查询的主要耗时）+ 本地 reranker；或用本机 Ollama 做 embedding：

```dotenv
RAG_EMBEDDING_PROVIDER=openai
RAG_EMBEDDING_BASE_URL=http://localhost:11434/v1     # Ollama 的 OpenAI 兼容接口，密钥可不填
RAG_EMBEDDING_MODEL=bge-m3
# RAG_RERANKER_PROVIDER=local（默认）
```

其他模板里给出的例子：阿里云百炼 DashScope 兼容模式的 embedding（`https://dashscope.aliyuncs.com/compatible-mode/v1`，模型 `text-embedding-v3`，批大小需设为 `10`——模板注释里说该服务文档标注的上限是 10，同样未验证）。

## 换 embedding：索引签名规则

**不同 embedding 产生的向量互不兼容**：维度不同会直接报错，维度相同则会悄悄检索出错误结果。所以：

- 入库时 ragbase 把 embedding **签名**（提供方、模型、维度）写到索引旁的 `.index_info.json`（位置由 `RAG_INDEX_INFO_FILE` 决定，默认与 `.ingest_meta.json` 同目录）。放文件而不是向量库元数据，是因为它与后端无关，`migrate-store` 之后仍然有效。文件内容形如：

  ```json
  {"embedding": {"provider": "local", "model": "Qwen/Qwen3-Embedding-0.6B", "dimensions": null},
   "dimension": 1024, "updated_at": "2026-10-17T14:50:00+00:00"}
  ```

  （`dimension` 是实际向量长度；示例数值仅示意。）
- 签名只含提供方、模型、**显式配置的**维度（本地模型忽略 `RAG_EMBEDDING_DIMENSIONS`）；地址、密钥、批大小不计入。
- `ragbase serve` / `ask` / `chat` / 增量 `ingest`（以及 `eval`、校准脚本）启动时比对，**不一致就拒绝**并提示“请运行 ragbase ingest --full 重建索引”；`ragbase doctor` 把它报为 `✗`。要继续用原索引，就把 embedding 配置改回建索引时的值。
- 只有 `ragbase ingest --full` 会改写签名（它先清空索引再重建）。入库时还会校验返回的向量长度与索引里已有向量一致。
- **没有该文件的旧索引**视为用默认本地模型建的：当前配置恰好是默认值就自动补记，否则拒绝。文件损坏同样拒绝（宁可拒绝也不猜）。

> 服务在线时**不要**运行 `ragbase ingest --full`（它会清空整个向量库，而服务进程还持有该索引）。换 embedding 的正确流程：停服务 → 改 `.env` → `ragbase ingest --full` → 启动服务。

## reranker 换模型后重新标定拒答阈值

`RAG_MIN_SCORE`（默认 `0.7`）和 `RAG_CONTEXT_MIN_SCORE`（默认 `0.95`）是按**默认本地 Qwen3-Reranker-0.6B 的 P(相关) 分布**标定的。别的 reranker（另一个模型或云端服务）分数分布不同，有的整体偏低，用同一个阈值会“总是拒答”或“从不拒答”。换 reranker 后请重新标定；`ragbase doctor` 在 reranker 不是默认本地模型而 `RAG_MIN_SCORE` 仍是 `0.7` 时会警告。`RAG_RERANKER_PROVIDER=none` 没有分数，无需标定。

### 校准脚本

```bash
uv run python scripts/calibrate_rerank_threshold.py [--mode hybrid]   # 只做召回 + 精排，不调用大模型
```

它对两组问题各取最相关片段（top1）的精排得分：

- **(a) 库内可回答**：`eval_questions.json` 里的全部题目 + 脚本内置的 15 道口语化补充题 `EXTRA_ANSWERABLE`——阈值应让它们全部通过；
- **(b) 库外问题**：脚本内置的 21 道 `OUT_OF_SCOPE`（分“无关 / 相邻 / 似是而非”三类）——阈值应尽量把它们拦下。

输出包括：每题得分；两组的 `min / median / max`；候选阈值（0.05 … 0.95）下的**误拒(a)**（可回答题被拒的数量）与**漏拦(b)**（库外题被放行的数量）；最后给出推荐值 = **可回答题最低分 × 0.75**（给没测到的题留余量），以及该阈值下拦下多少库外题、哪些库外题与可回答题分数重叠。把推荐值写进 `RAG_MIN_SCORE`，必要时同时调整 `RAG_CONTEXT_MIN_SCORE`。云端 reranker 每个问题会调一次 rerank 接口（共约 `题数 + 15 + 21` 次，默认评测集 30 题时约 66 次，会产生少量费用）。

> **重要**：脚本里的评测题和库外问题是针对**作者自己的语料**（《系统架构设计师》《信息系统项目管理师》教材）写的。用在你自己的资料上时，要把 `eval_questions.json`（放在 `RAG_HOME` 下）换成你的题目，并改掉脚本里的 `EXTRA_ANSWERABLE` / `OUT_OF_SCOPE`，否则得到的数字没有意义。`eval_questions.json` 的格式见 [evaluation.md](evaluation.md#评测题格式)。

### 默认阈值是怎么来的（本地 Qwen3-Reranker-0.6B）

数据来自作者的教材语料，记录在 `config.py` 的注释里（旧数据，供理解数字含义，不是你的语料的结论）：

- 45 道库内问题的 top1 得分 0.937 ~ 0.9999（中位 0.9994）；21 道库外问题 0.0000 ~ 0.9882——其中 14 道低于 0.06，其余 0.61 ~ 0.99 与库内分数有重叠。
- 取“最低库内分 × 0.75 ≈ 0.7”：库内零误拒，拦下 15 / 21；**宁可放过也不误拒**。得分重叠的库外题靠阈值拦不住，要靠提示词里的“只依据片段、没有依据就说没有”兜底。
- `RAG_CONTEXT_MIN_SCORE`：本地 reranker 的得分高度饱和（库内题 Top5 的得分基本都 ≥ 0.96），30 道评测题上的实验显示阈值 ≤ 0.95 对库内题等于不过滤，0.97 起开始损失要点召回（平均保留片段 5.00 → 4.87，`point_recall` 0.8344 → 0.8210）。所以取 0.95，只在片段得分明显偏低时才裁剪；调高会开始丢掉对比 / 多方面题需要的片段。

## 硬件与速度

| 设备 | 说明 |
| --- | --- |
| Apple Silicon（MPS） | 自动选择，模型用 fp16；作者的开发和日常使用环境，服务启动约 10 秒（加载两个本地模型和 BM25 索引） |
| NVIDIA GPU（CUDA） | 自动选择，fp16（代码支持，**没有在 CUDA 机器上实测过**） |
| 纯 CPU | 可以运行（Docker 镜像就是 CPU 版），但精排和批量向量化明显更慢：每个问题的精排要对约 20 个候选片段逐一打分；首次建索引的大 PDF 会很久。`doctor` 会对此给出警告。具体耗时取决于机器，这里不给数字 |

用 `RAG_DEVICE=auto|mps|cuda|cpu` 指定设备。磁盘：两个本地模型约 2.2 GB，加上 Python 依赖（含 PyTorch）数 GB。

## 国内网络：Hugging Face 镜像

模型从 Hugging Face 下载。访问慢或不通时设置镜像端点（`models download` 和 `serve --download-models` 都遵循它）：

```bash
HF_ENDPOINT=https://hf-mirror.com uv run ragbase models download
# 或写进 .env：HF_ENDPOINT=https://hf-mirror.com
```

- 模型缓存位置跟随 `HF_HOME`（默认 `~/.cache/huggingface`）；已缓存的模型会跳过，下载中断后重跑可续传。
- `ragbase serve` 在**索引非空而模型缺失**时会提示先运行 `ragbase models download`（而不是抛一长串报错）；索引为空时服务会照常启动（本来就不加载模型）；加 `--download-models` 则缺什么先下什么。
- 也可以在能访问 Hugging Face 的机器上下载后，把整个 `HF_HOME` 目录（或 Docker 的 `ragbase-models` 卷）拷过来。

## 大模型

- 走 OpenAI 兼容协议：OpenAI、DeepSeek、通义、火山方舟、Moonshot、本地 Ollama / vLLM 等，只要换成该服务商给的地址、密钥、模型名（模板里的地址只是示例，未逐一验证）。
- 对话 Agent 需要模型支持 `tools` / `tool_calls`（函数调用）。单元测试用的是脚本化假模型，**没有用真实大模型验证过**，各家服务商对工具调用的支持需要自己确认。
- **模型 ID 的写法因服务商而异**（例如火山方舟的 ID 用连字符并带日期后缀，推理接入点则是 `ep-` 开头）。写错通常表现为 404 / 400；不确定时先列出服务商支持的模型：`curl -H "Authorization: Bearer $KEY" {BASE_URL}/models`（大多数 OpenAI 兼容服务提供该接口，是否可用取决于服务商）。更多见 [troubleshooting.md](troubleshooting.md#模型接口返回-4xx--模型端点已关闭--模型-id-写错)。
- 验证配置：`ragbase doctor --check-llm`（发一次最小请求）；云端 embedding / rerank 用 `--check-models`。两者都会产生极少量 API 调用。
- 评测可以用另一个裁判模型，见 `RAG_EVAL_LLM_*` 与 [evaluation.md](evaluation.md)。
