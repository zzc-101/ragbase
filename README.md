# RAG Demo

面向中文 PDF、Word、Markdown 和文本资料的本地 RAG 命令行问答 Demo。

## 架构

```text
文档 → 解析 → 中文切分 → Qwen3-Embedding-0.6B → ChromaDB
                                                      ↓
提问 → 向量召回 Top 20 → Qwen3-Reranker 精排 Top 5 → 豆包生成 → 答案与来源
```

- Embedding 和 Reranker 在本机运行，默认使用 Apple Silicon 的 MPS。
- 回答模型通过 OpenAI 兼容接口调用火山引擎方舟。
- 向量库、文档和密钥均已在 `.gitignore` 中排除。

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

结果写入 `eval_results.json`。评测会调用回答模型，因此会产生 API 费用。

## 关键参数

参数目前直接定义在代码中：

- 切分大小与重叠：[rag_common.py](rag_common.py)
- 召回和精排数量：[query.py](query.py)、[evaluate.py](evaluate.py)
- 本地模型与设备：[embed.py](embed.py)、[reranker.py](reranker.py)

扫描版 PDF 或文本层质量较差的 PDF 需要另行增加 OCR；当前项目不会自动 OCR。

## 常见问题

- `AuthenticationError`：检查 `.env` 中的方舟 API Key、模型接入点和地址。
- 找不到向量库：先运行 `uv run ingest.py`。
- MPS 不可用：把 `embed.py` 和 `reranker.py` 中的 `device` 改为 `cpu`。
- 依赖环境异常：重新执行 `uv pip sync requirements.lock`。
