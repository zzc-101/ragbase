# ragbase

面向中文 PDF、Word、Markdown 和文本资料的本地 RAG 知识库：混合检索 + 精排、命令行问答 / 对话 Agent、HTTP 服务与 Web 界面、评测。命令行入口为 `ragbase`。

## 快速开始

你需要准备三样东西：

1. **一个 OpenAI 兼容的大模型服务**（OpenAI、DeepSeek、通义、火山方舟、本地 Ollama / vLLM 等都行）：地址 `RAG_LLM_BASE_URL`、密钥 `RAG_LLM_API_KEY`、模型名 `RAG_LLM_MODEL`。不配也能启动和检索，只是不能问答 / 对话。
2. **约 2.2 GB 磁盘放本地模型**（Qwen3-Embedding-0.6B 与 Qwen3-Reranker-0.6B 各约 1.1 GB，首次运行下载一次），外加 Python 依赖（含 PyTorch）数 GB。
3. **你的资料**：PDF / `.docx` / Markdown / 纯文本。

三条路径任选其一，最终效果一样：浏览器打开 <http://127.0.0.1:8000/>。

### 路径 A：Docker Compose（CPU 版，不用装 Python / Node）

```bash
cp .env.example .env              # 编辑 .env：填 RAG_LLM_*；对外开放服务必须设置 RAG_API_KEY（见下方“安全提示”）
mkdir -p ragbase-home/data        # 项目主目录（挂载为容器内 /data）；把文档放进 ragbase-home/data/
docker compose run --rm ragbase models download   # 首次：下载模型到 ragbase-models 卷（约 2.2 GB，之后复用）
docker compose run --rm ragbase ingest            # 建立索引
docker compose up -d --build                      # 启动，浏览器打开 http://127.0.0.1:8000/
docker compose run --rm ragbase doctor            # 随时体检
```

- 国内网络下载模型慢或不通：`HF_ENDPOINT=https://hf-mirror.com docker compose run --rm ragbase models download`（也可以把 `HF_ENDPOINT=https://hf-mirror.com` 写进 `.env`，compose 会自动读取）。
- 镜像是 **CPU 版**（`RAG_DEVICE=cpu`），不含模型权重；`docker compose up` 的启动命令带 `--download-models`，模型缺失时会自动下载，所以第 3 步也可以省略（但下载期间服务还没起来）。
- 新增文档：服务在线时到 Web 界面“知识库”页上传（在线增量入库；前提是启动时索引非空）；或者 `docker compose stop ragbase` → 把文件放进 `ragbase-home/data/` → `docker compose run --rm ragbase ingest` → `docker compose up -d`（`ingest` 不要与运行中的服务同时跑）。
- 端口 / 监听：默认只绑定 `127.0.0.1:8000`。要让局域网访问，设置 `RAGBASE_BIND=0.0.0.0`（可选 `RAGBASE_PORT`）——**务必同时设置 `RAG_API_KEY`**。
- Linux 上 `ragbase-home` 要由你自己先 `mkdir`（容器里是 uid 1000 的非 root 用户）；不想管权限可在 `docker-compose.yml` 里改用命名卷。
- 可选 Qdrant 服务端做向量库：在 `.env` 加 `RAG_VECTOR_STORE=qdrant` 与 `RAG_QDRANT_URL=http://qdrant:6333`，用 `docker compose --profile qdrant up -d` 启动；已有 Chroma 索引可 `docker compose run --rm ragbase migrate-store --to qdrant` 迁移。
- 注意：Docker 相关文件在没有 Docker daemon 的环境里只做过 `docker compose config` 静态校验，镜像是否能成功构建 / 运行请以你本机为准。

### 路径 B：`scripts/start.sh`（克隆仓库后一键，macOS / Linux）

```bash
git clone <本仓库地址> ragbase && cd ragbase
scripts/start.sh                  # 国内网络：HF_ENDPOINT=https://hf-mirror.com scripts/start.sh
```

脚本依次：检查 `uv`（没有会打印安装命令）→ `uv sync --locked` → 构建 Web 界面（`web/dist` 不存在且装了 `pnpm` 时）→ 没有 `.env` 就运行 `ragbase init` → `ragbase models download` → `ragbase doctor` → `ragbase serve`。可重复运行，已完成的步骤会跳过。额外参数会传给 `ragbase serve`（如 `scripts/start.sh --host 0.0.0.0 --port 8080`）；`--no-serve` 只做准备、不启动。首次运行后请编辑 `.env` 填入大模型配置，把文档放进 `data/`，另开终端运行 `uv run ragbase ingest`。

### 路径 C：手动

```bash
uv sync                           # 需要 Python >= 3.12 和 uv（https://docs.astral.sh/uv/）
uv run ragbase init               # 生成 data/、.env、bm25_userdict.txt（不会覆盖已有文件）
$EDITOR .env                      # 填 RAG_LLM_BASE_URL / RAG_LLM_API_KEY / RAG_LLM_MODEL
uv run ragbase models download    # 下载 embedding + reranker（已缓存则跳过）
cp ~/你的资料/*.pdf data/          # 放文档
uv run ragbase ingest             # 建立索引（首次会向量化全部文档，耗时与资料量有关）
uv run ragbase serve              # http://127.0.0.1:8000/
```

Web 界面需要前端构建产物：仓库里运行 `scripts/build_web.sh`（Node 22 + pnpm）；没有构建时 API 仍可用，页面会提示“前端未构建”。

### 体检：`ragbase doctor`

一屏检查 Python 版本、目录是否可写、`.env`、计算设备、模型是否已缓存、向量库是否可访问及片段数、是否需要 `ragbase ingest`、大模型是否配置（只显示协议 / 模型 / 地址，不显示密钥）、接口鉴权、Web 资源，每项给出 `✓`（正常）/ `!`（警告）/ `✗`（必须修复）和修复提示；有 `✗` 时退出码非零。默认**不调用大模型**；加 `--check-llm` 才会发一次最小请求验证地址 / 密钥 / 模型名。

### 国内网络：Hugging Face 镜像

模型从 Hugging Face 下载。访问慢或不通时设置镜像端点（`models download` 和 `serve --download-models` 都遵循它）：

```bash
HF_ENDPOINT=https://hf-mirror.com uv run ragbase models download
# 或写进 .env：HF_ENDPOINT=https://hf-mirror.com
```

模型缓存位置跟随 `HF_HOME`（默认 `~/.cache/huggingface`）；已缓存的模型会跳过，下载中断后重跑可续传。`ragbase serve` 在**索引非空而模型缺失**时会提示先运行 `ragbase models download`（而不是抛一长串报错）；加 `ragbase serve --download-models` 则缺什么先下什么。

### 换模型

默认模型可通过 `RAG_EMBEDDING_MODEL`（默认 `Qwen/Qwen3-Embedding-0.6B`）和 `RAG_RERANKER_MODEL`（默认 `Qwen/Qwen3-Reranker-0.6B`）更换，`ragbase models download` / `doctor` 会跟着用新名字。

- **更换 embedding 模型会改变向量维度，必须重新入库**：`ragbase ingest --full`（旧索引与新向量不兼容）。
- **更换 reranker 后得分分布会变，需要重新校准 `RAG_MIN_SCORE`**（拒答阈值，默认 0.7 是按 Qwen3-Reranker-0.6B 校准的），可用 `scripts/calibrate_rerank_threshold.py`；`RAG_CONTEXT_MIN_SCORE` 同理。
- 目前代码按 Qwen3 系列的用法编写（查询 prompt、yes/no logit 打分）；换成别家模型不保证可用。

### 硬件与速度

| 设备 | 说明 |
| --- | --- |
| Apple Silicon（MPS） | 自动选择，模型用 fp16；开发和日常使用的主要环境，服务启动约 10 秒 |
| NVIDIA GPU（CUDA） | 自动选择，fp16（代码支持，本次改动未在 CUDA 机器上实测） |
| 纯 CPU | 可以运行（Docker 镜像就是 CPU 版），但精排和批量向量化明显更慢（每个问题的精排要对约 20 个候选片段逐一打分）；首次建索引的大 PDF 会很久。`doctor` 会对此给出警告 |

用 `RAG_DEVICE=auto|mps|cuda|cpu` 指定设备。CPU 的具体耗时取决于机器，这里不给数字。

### 数据放在哪里（项目主目录 `RAG_HOME`）

`RAG_HOME` 环境变量指定项目主目录，未设置则为**当前工作目录**；`ragbase init [DIR]` 在这里生成初始文件。Docker 里它是 `/data`（挂载自 `./ragbase-home`）。

```text
<RAG_HOME>/
  .env                  配置（RAG_* 变量；含密钥，别提交）
  data/                 你的文档（不递归子目录）
  chroma_db/            向量索引（RAG_VECTOR_STORE=chroma 时）
  qdrant_db/            向量索引（qdrant 本地模式时）
  .ingest_meta.json     增量入库的文件指纹
  .bm25_cache/          BM25 分词缓存
  bm25_userdict.txt     领域词典（jieba 格式）
  web/dist/             前端构建产物（没有则回退到包内自带的副本）
~/.cache/huggingface/   本地模型缓存（HF_HOME；Docker 里是 /models 卷）
```

### 安全提示

服务一旦监听 `0.0.0.0`（Docker 里默认如此）或暴露到公网，**必须设置 `RAG_API_KEY`**，否则任何能访问端口的人都能查询你的知识库、消耗你的大模型额度、上传 / 删除文档。生成密钥：`ragbase init --api-key`（新建 `.env` 时自动写入）或 `openssl rand -hex 32`。细节与 HTTPS 建议见“接口鉴权”。

## 架构

```text
文档 → 解析 → 清洗 → 中文切分 → Qwen3-Embedding-0.6B → 向量库（ChromaDB 默认 / Qdrant 可选）
                                                      ↓
提问 → 向量召回 + BM25 召回 → RRF 融合 Top 20 → Qwen3-Reranker 精排 Top 5 → 豆包生成 → 答案与来源
```

- Embedding 和 Reranker 在本机运行，设备自动选择（Apple Silicon 的 MPS > CUDA > CPU，可用 `RAG_DEVICE` 指定）。
- 两个本地模型会优先从本机缓存加载；重复问题的查询向量和精排分数会在进程内缓存。
- 回答模型通过 OpenAI 兼容接口调用火山引擎方舟。
- 向量库、文档和密钥均已在 `.gitignore` 中排除。

入库预处理按职责拆分为（均在 `src/ragbase/ingest/` 下）：

- `parser.py`：文件格式解析
- `cleaner.py`：文本规范化、重复页眉页脚和空页清理
- `splitter.py`：递归切分、空片段与重复片段过滤
- `toc.py`：PDF 书签 -> 每页章节路径（`heading_path`）
- `common.py`：稳定 ID、文件指纹和入库编排
- `service.py`：增量入库逻辑（命令行和 HTTP 服务共用，依赖全部显式传入，便于用临时目录 + 假 embedding 测试）
- `jobs.py`：HTTP 服务里的后台入库任务

## 项目结构

```text
pyproject.toml / uv.lock       依赖与锁文件（uv 管理）；命令行入口 ragbase = ragbase.cli:main
src/ragbase/
  cli.py  __main__.py          命令行：init / models / doctor / serve / ingest / ask / chat / eval / migrate-store（python -m ragbase 等价）
  scaffold.py  doctor.py       ragbase init（生成 .env 等）、ragbase doctor（环境自检）
  resources/                   随包分发的模板：env.example（.env 模板）、bm25_userdict.txt、data_readme.txt
  web/dist/                    （可选，已 gitignore）随包分发的前端构建产物，scripts/build_web.sh 生成
  config.py  logging_setup.py  集中配置（RAG_* 环境变量 / .env）；日志
  llm/factory.py               大模型工厂（OpenAI 兼容协议）
  models/                      embeddings / reranker（本地 Qwen3）、download（缓存检查与下载）
  ingest/                      parser / cleaner / splitter / toc / common / service / jobs
  vectorstores/                向量库抽象（base）、chroma、qdrant、factory（按 RAG_VECTOR_STORE 创建）、migrate（后端间迁移）
  retrieval/                   hybrid（向量 + BM25 + RRF）、pipeline（召回 -> 精排 -> 拒答 -> 生成）、labels（来源标签）
  agent/                       tools（检索工具与引用登记）、service（对话 Agent 与会话）
  api/                         app（FastAPI 应用，ragbase.api.app:app）、schemas、auth（API Key 鉴权）
  evaluation/                  runner（评测流程）、metrics（Ragas 与词面指标）
scripts/                       start.sh（一键启动）、build_web.sh（构建前端并复制进包）、拒答阈值校准
Dockerfile / docker-compose.yml  CPU 镜像与 Compose 部署（.env.example 是配置模板）
tests/                         断言脚本式单元测试（tests/run_all.py 一次运行全部）
web/                           前端工程（React + Vite）
data/ chroma_db/ qdrant_db/ ...  运行时数据，默认都在项目主目录下（见下）
```

**项目主目录**：环境变量 `RAG_HOME`，未设置则为**当前工作目录**。`data/`、`chroma_db/`、`qdrant_db/`、`.ingest_meta.json`、`.bm25_cache/`、`web/dist`、`bm25_userdict.txt`、`eval_questions.json`、`eval_results.*.json`、`.ragas_cache/` 以及 `.env` 的默认位置都从它推导，所以请在仓库根目录运行命令（或设置 `RAG_HOME` 指向它）。`RAG_HOME` 必须是真正的环境变量，不能写在 `.env` 里（`.env` 的位置本身依赖它）。

## 混合检索

[src/ragbase/retrieval/hybrid.py](src/ragbase/retrieval/hybrid.py) 同时走两路召回：向量路直接查向量库（Chroma 或 Qdrant，各取 30），BM25 路用 `jieba` 分词 + `bm25s` 打分（各取 30），再用 RRF（`score = Σ 1/(60+rank)`，只看排名）融合取前 20 交给 Reranker。专有名词、缩写和公式（如 TOGAF、SPI）这类向量容易漏的内容，由 BM25 补召回。

- BM25 索引在启动时从向量库已有的片段文本现建（按片段 ID 排序后分词，所以换向量库后端分词缓存仍然有效），无需改动入库流程。
- 分词结果缓存在 `.bm25_cache/`（已忽略）；缓存键包含全部片段内容的摘要和用户词典，入库 / 删除 / 章节标题变化后自动失效，旧缓存文件会被清掉。
- BM25 索引、语料和书目录打包成不可变的“快照”，入库或删除文档后在后台建好新快照，再整体替换一个引用（原子切换），检索期间不会看到半成品索引。
- 领域词典放在项目主目录（仓库根目录）的 [bm25_userdict.txt](bm25_userdict.txt)（jieba 格式：`词 词频 词性`）。不要放进 `data/`，那里的 `.txt` 会被当作文档入库。修改后下次启动自动生效。
- 召回模式用 `--mode` 切换：`uv run ragbase ask --mode {vector,bm25,hybrid}`（默认 `hybrid`）。答案后的来源会标出召回路径，如 `[V3+B1]` 表示向量路第 3、BM25 路第 1。
- 单元测试：`uv run python tests/test_hybrid.py`。

## 章节标题（PDF 书签）

入库时读取 PDF 书签（目录），给每个片段加上 `heading_path`（如 `第11章 项目成本管理 > 11.3 挣值分析`），见 [src/ragbase/ingest/toc.py](src/ragbase/ingest/toc.py)：

- 对第 p 页，路径 = 起始页不晚于该页的各层级“最近一条”书签标题，用 ` > ` 连接；标题会去首尾空白、压缩空白，纯数字 / 标点的标题（如考试大纲里的 `1`）直接丢弃，没有可用书签则该文件没有章节。非 PDF 暂无章节。
- **近似**：书签只记录起始页，不记录页内位置，一节从页中间开始时，整页都归给这一节。
- 向量化输入 = `heading_path + "\n" + 正文`，BM25 分词输入 = `heading_path + " " + 正文`；向量库里存的 document 仍是纯正文。**精排 Reranker 的输入只用正文**，这样按纯正文校准的拒答阈值仍然有效。
- 展示：`/search` 的 `chunks`、`/ask` 的 `citations` 带 `heading_path`（没有为 `null`）；交给大模型的上下文写成 `[片段1]（第11章 … > 11.3 …）正文`；`ragbase ask` 的来源标签带末级章节，如 `xxx.pdf:第12页「11.3 挣值分析」[V2+B1|0.93]`。
- `INGEST_PIPELINE_VERSION` 已升到 `3`、分词版本升到 `3`：下次增量入库会把所有文件视为已变化并重新索引（这才会给现有的 PDF 加上章节）。**旧索引（没有 `heading_path`）无需改动即可继续使用**，章节全部为 `null`。

## 低相关度拒答

精排后，如果最相关片段的 `rerank_score` 低于阈值，直接回答“我没有在文档中找到相关内容。”，**不调用大模型**（省钱，也避免模型凭空编造）。

- `rerank_score` 是 Reranker 给出的 P(相关)，范围 0~1（模型原始输出是 logit 差，经 sigmoid 转换）。每个返回片段的 metadata 里都有，命令行来源会显示成 `xxx.pdf:第12页[V2+B1|0.93]`。
- 默认阈值是 [src/ragbase/config.py](src/ragbase/config.py) 的 `RERANK_MIN_SCORE`（也可用 `RAG_MIN_SCORE` 环境变量覆盖）；临时调整用 `--min-score`，`0` 表示不拒答：

```bash
uv run ragbase ask --min-score 0.3
uv run ragbase eval --retrieval-only --min-score 0.3   # 同时统计会被拒答的题数
```

- 重新校准（换了教材、Reranker 或召回方式后）：运行下面的脚本，不调用 LLM。它会对 30 道评测题和一组库外问题打印最相关片段的得分分布，以及各候选阈值的“误拒 / 漏拦”数，再把推荐值填回 `RERANK_MIN_SCORE`。需要拦截更多库外问题时可以调高阈值，代价是更容易误拒正常问题。

```bash
uv run python scripts/calibrate_rerank_threshold.py
```

- 单元测试：`uv run python tests/test_refusal.py`。

## 抑制幻觉：片段过滤与“先摘录再回答”

**1. 只把足够相关的片段交给大模型。** 通过拒答门槛后，只保留 `rerank_score >= CONTEXT_MIN_SCORE` 的片段（最相关的 top1 总是保留），其余不进入提示词。默认值 `0.95` 定义在 [src/ragbase/config.py](src/ragbase/config.py)（环境变量 `RAG_CONTEXT_MIN_SCORE`），旁边的注释记录了 30 道评测题的阈值实验。目前 Reranker 得分很饱和（库内题 Top5 的得分基本都 ≥ 0.96），所以该阈值对库内题几乎不裁剪，主要是兜底；调高会开始丢掉对比 / 多方面题需要的片段。临时调整：

```bash
uv run ragbase ask --context-min-score 0.99      # 0 表示不过滤
uv run ragbase eval --retrieval-only --context-min-score 0.99
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
uv sync            # 按 uv.lock 创建 .venv 并以可编辑模式安装 ragbase（Linux 自动用 CPU 版 torch，macOS 用 PyPI 版以支持 MPS）
```

之后用 `uv run ragbase ...`（或直接 `.venv/bin/ragbase ...`）运行命令；`ragbase --help` 查看全部子命令。依赖写在 `pyproject.toml`，升级后用 `uv lock` 更新 `uv.lock`（`constraint-dependencies` 钉住了传递依赖的版本，要升级某个传递依赖时先改这里）。

创建 `.env`：运行 `uv run ragbase init` 会按模板 [.env.example](.env.example)（全部配置项分组、带中文注释，示例都已注释掉）生成，也可手动 `cp .env.example .env`。最少只需要大模型这三项：

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
uv run ragbase ingest --full
```

以后文档新增、修改或删除时：

```bash
uv run ragbase ingest
```

全量入库和增量入库使用相同的稳定片段 ID，并通过 `.ingest_meta.json` 记录文件状态（每个文件成功后立即写入，中途失败不会重做已完成的文件；失败的文件下次重试）。旧版索引首次执行增量命令时会迁移重建：先按 ID 分批清空旧片段，再逐文件写入（不删除集合）。

> **服务运行期间不要运行 `ragbase ingest --full`**：它会清空整个向量库（删除集合并重建），而服务进程还持有该索引。服务在线时请用下面的 `POST /documents` / `POST /ingest` / `DELETE /documents/{name}`（或先停服务再跑命令行）。`ragbase ingest`（增量）同样不要与服务同时运行（两个进程会同时写同一个库）。

## 向量库

向量库是可插拔的：默认 **Chroma**（本地目录），可切换为 **Qdrant**（本地嵌入式目录或 Qdrant 服务端）。入库、检索、API、迁移都只依赖 [src/ragbase/vectorstores/base.py](src/ragbase/vectorstores/base.py) 里的 `VectorStore` 接口，换后端不需要改任何业务代码；混合检索的 RRF 只看排名，所以两个后端的距离 / 相似度数值是否可比并不重要，只要排序正确。

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAG_VECTOR_STORE` | `chroma` | 向量库后端：`chroma` / `qdrant`；其他取值启动时报错 |
| `RAG_COLLECTION` | `rag` | 集合名（Chroma collection / Qdrant collection） |
| `RAG_DB_DIR` | 主目录下 `chroma_db/` | Chroma 数据目录（仅 `chroma`） |
| `RAG_QDRANT_PATH` | 主目录下 `qdrant_db/` | Qdrant **本地嵌入式**模式的数据目录（`RAG_QDRANT_URL` 未设置时使用） |
| `RAG_QDRANT_URL` | 无 | Qdrant **服务端**地址，如 `http://localhost:6333`；设置后忽略 `RAG_QDRANT_PATH` |
| `RAG_QDRANT_API_KEY` | 无 | Qdrant 服务端的 API Key（`SecretStr`，不会出现在日志 / `repr` 中） |

**切换步骤**（不想重新向量化的话）：先停服务，再迁移，最后改配置并重启：

```bash
uv run ragbase migrate-store --to qdrant            # 从 chroma 复制到 qdrant（--from 默认是另一个后端）
echo 'RAG_VECTOR_STORE=qdrant' >> .env              # 然后重启 ragbase serve
```

`migrate-store` 把 ID、向量、正文和元数据按批（`--batch-size`，默认 500）从源库复制到目标库，**不加载模型、不重新向量化**，结束后校验源库 / 已复制 / 目标库的片段数一致。目标库非空时拒绝，加 `--force` 会先清空目标库再复制。片段 ID 与正文不变，所以 `.ingest_meta.json` 和 `.bm25_cache/` 继续有效，迁移完成后增量入库照常工作；反向迁移同理（`--to chroma`）。迁移时请停掉服务（本地 Qdrant 目录同一时刻只允许一个进程打开，见下）。

**本地模式 vs 服务端模式**

- 本地模式（只设 `RAG_VECTOR_STORE=qdrant`）：数据在 `qdrant_db/`，不用另外部署服务，适合单机；过滤和检索在进程内存里完成（几千到十万级片段够用），不支持 payload 索引。**同一个目录同一时刻只允许一个进程打开**：`ragbase serve` 运行时再运行 `ragbase ingest` / `ragbase ask` / `migrate-store` 会报“already accessed by another instance”——请用服务的在线入库接口，或先停服务。进程内同一目录共享一个 client（线程安全，调用被锁串行化）。
- 服务端模式（再设 `RAG_QDRANT_URL`）：连接独立运行的 Qdrant（如 `docker run -p 6333:6333 qdrant/qdrant`），支持多进程同时访问和 payload 索引（`fid`、`source` 建了 keyword 索引）。
- 实现细节：Qdrant 点 ID 必须是 UUID / 整数，字符串片段 ID 用 `uuid5(固定命名空间, chunk_id)` 映射，原始 ID、正文和元数据存在 payload（`chunk_id` / `document` / `metadata`）；距离用余弦（Qdrant 会把向量归一化，Qwen3 向量本来就是归一化的）；集合在第一次写入时按向量维度创建。
- 两者在 upsert 同 ID 时对元数据的处理略有不同（Chroma 按键合并，Qdrant 整体替换）；入库每次都写完整元数据，不受影响。

## 问答与评测

启动交互式问答：

```bash
uv run ragbase ask
```

输入 `exit`、`quit` 或 `q` 退出。

运行 30 道内置评测题：

```bash
uv run ragbase eval
```

评测包含三层指标：

- Ragas：忠实度、回答相关性、事实正确性、上下文精确率和召回率
- 无需 LLM 的词面代理指标：要点召回、上下文精确率和 MRR（分别统计召回 Top20、精排 Top5 和实际交给 LLM 的片段）
- 原有中文 LLM 裁判：回答正确性与检索充分性

先用一道题冒烟测试：

```bash
uv run ragbase eval --limit 1
```

只运行词面指标和问答生成、跳过额外的评测模型调用：

```bash
uv run ragbase eval --skip-ragas --skip-legacy-judge
```

对比召回方式（`--retrieval-mode {vector,bm25,hybrid}`，默认 `hybrid`），结果写入 `eval_results.<mode>.json`。加上 `--retrieval-only` 则只跑召回 + 精排和词面指标，不调用任何 LLM，没有 API 费用（结果写入 `eval_results.<mode>.retrieval.json`）：

```bash
uv run ragbase eval --retrieval-only --retrieval-mode vector
uv run ragbase eval --retrieval-only --retrieval-mode hybrid
```

结果逐题写入对应的结果文件，Ragas 调用缓存于 `.ragas_cache/`。评测会调用回答模型，因此会产生 API 费用。当前测试集的 `answer_keyword` 会作为参考答案；后续可以为题目增加更完整的 `reference` 字段以提高评测可信度。

> 词面指标是基于参考答案要点的代理指标，不是基于人工相关文档标注的标准 Precision/Recall/MRR。

## 服务化运行

同一份代码也可以作为 HTTP 服务运行（FastAPI，[src/ragbase/api/app.py](src/ragbase/api/app.py)），`ragbase ask` / `ragbase eval` 仍可独立使用。

```bash
uv run ragbase serve --host 127.0.0.1 --port 8000   # 不带参数时读取 RAG_HOST / RAG_PORT；也可：uv run uvicorn ragbase.api.app:app
```

- **必须单进程，不要加 `--workers`**：embedding / reranker / BM25 索引都加载在服务进程内，多 worker 会让每个进程各占一份显存，且 MPS 模型不能并发推理。并发请求由线程池接住，召回 + 精排在一把锁内串行，大模型生成在锁外并行（慢生成不会阻塞检索）。
- 启动约 10 秒（加载两个本地模型和 BM25 索引）。索引不存在或为空时服务仍会启动，`/health` 报告 `not_ready`，`/search`、`/ask` 以及文档管理接口返回 503（“索引不存在或为空，请先运行 ragbase ingest --full”）；用命令行建好索引后需重启服务（在线入库要求服务启动时索引已存在）。
- 每个响应带 `X-Request-ID`（请求头里带了就沿用，否则生成），每个请求输出一行访问日志（方法、路径、状态码、耗时；`/search`、`/ask` 另带 `refused` 和 `top_score`）。
- 错误统一为 `{"error": "...", "detail": "..."}`：422 参数校验、400 请求了未构建的 BM25 模式 / 未知书名 / 非法上传、404 找不到文档或任务、409 已有入库任务在运行、413 上传文件过大、502 大模型调用失败（鉴权、限流、5xx 等，已按 `RAG_LLM_MAX_RETRIES` 重试过）、504 大模型响应超时（`error` 为 `llm_timeout`，单次请求超过 `RAG_LLM_TIMEOUT` 且重试用完）、503 索引 / 大模型未就绪、500 未处理异常（详情只写日志）。
- 交互式文档：<http://127.0.0.1:8000/docs>。

| 接口 | 说明 |
| --- | --- |
| `GET /health` | `status`（`ok` / `not_ready`）、`chunk_count`、`bm25_ready`、`bm25_built_at`（当前 BM25 索引的构建时间，UTC）、`bm25_chunk_count`（BM25 覆盖的片段数，与 `chunk_count` 不同说明刷新未完成）、`device`、`mode`、`llm_configured`、`vector_store`（向量库后端：`chroma` / `qdrant`）、`llm_protocol`、`llm_model`（模型名，不含密钥）、`agent_ready`（对话 Agent 是否就绪） |
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

**按书过滤**：`/search` 与 `/ask` 的 `sources` 是书名（文件名，取自 `GET /documents` 的 `name`）列表，只在这些书里检索；缺省或空列表表示全部；有未知书名返回 400 并列出可选项。服务端用检索器内存里的语料把书名解析成库里存的 `source` 原始值（旧索引存的是绝对路径，不会自己拼路径）；向量路让向量库按 `source` 过滤（Chroma 用 `where={"source": {"$in": [...]}}`，Qdrant 用 `MatchAny`），BM25 路用 bm25s 的 `weight_mask`，过滤后没有命中则判定拒答。命令行：`uv run ragbase ask --source 书名.pdf`（可重复）。

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
- 旧版索引（片段无 `fid`、ID 是 UUID）第一次 `POST /ingest` 或上传时会迁移：按 ID 分批清空旧片段（不删除集合）再逐文件重建，耗时与全量入库相当。`DELETE` 对旧片段按 `source` 的文件名匹配。

**配置**（环境变量或 `.env`，环境变量优先；[src/ragbase/config.py](src/ragbase/config.py) 是唯一来源，`ragbase ask` / `ragbase eval` 的默认值也取自这里，命令行参数仍可覆盖）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RAG_DEVICE` | `auto` | 本地模型设备：`auto`（mps > cuda > cpu）/ `mps` / `cuda` / `cpu` |
| `RAG_MODE` | `hybrid` | 默认召回方式 |
| `RAG_RETRIEVE_K` | `20` | 精排前的候选数 |
| `RAG_PER_ROUTE_K` | `30` | 混合检索每一路各取多少 |
| `RAG_RERANK_K` | `5` | 精排后保留的片段数 |
| `RAG_MIN_SCORE` | `0.7` | 拒答阈值，0 表示不拒答 |
| `RAG_CONTEXT_MIN_SCORE` | `0.95` | 交给大模型的片段最低精排得分，0 表示不过滤 |
| `RAG_HOME`（须为真正的环境变量） | 当前工作目录 | 项目主目录：下面各路径的默认位置、`.env`、`bm25_userdict.txt`、`eval_questions.json` 都在它下面 |
| `RAG_DATA_DIR` / `RAG_DB_DIR` | 主目录下 `data/` / `chroma_db/` | 文档目录 / Chroma 持久化目录（命令行脚本和服务共用） |
| `RAG_VECTOR_STORE` / `RAG_COLLECTION` | `chroma` / `rag` | 向量库后端与集合名，详见下面的“向量库”一节 |
| `RAG_META_FILE` / `RAG_CACHE_DIR` | 主目录下 `.ingest_meta.json` / `.bm25_cache/` | 入库元数据文件 / BM25 分词缓存目录 |
| `RAG_MAX_UPLOAD_MB` | `300` | `POST /documents` 单文件大小上限 |
| `RAG_AGENT_MAX_TOOL_CALLS` | `4` | 对话 Agent 每轮用户消息最多调用几次工具 |
| `RAG_AGENT_MAX_MODEL_CALLS` | `6` | 对话 Agent 每轮用户消息最多调用几次大模型（至少 2） |
| `RAG_CHAT_HISTORY_TURNS` | `10` | 每个会话保留最近几轮对话 |
| `RAG_CHAT_MAX_SESSIONS` | `200` | 内存中最多保留的会话数 |
| `RAG_CHAT_SESSION_TTL` | `3600` | 会话空闲多少秒后过期 |
| `RAG_LOG_LEVEL` | `INFO` | 日志级别 |
| `RAG_HOST` / `RAG_PORT` | `127.0.0.1` / `8000` | `ragbase serve` 的监听地址 |
| `RAG_API_KEY` | 无（不鉴权） | 接口共享密钥；设置后所有 API 请求都要带它，见“接口鉴权” |
| `RAG_CORS_ORIGINS` | 无（不加 CORS） | 允许跨域的前端来源，逗号分隔，见“接口鉴权” |

**大模型配置**（`llm_protocol` 目前只实现 `openai`，即 OpenAI 兼容的 `/chat/completions`；新增协议只需在 [src/ragbase/llm/factory.py](src/ragbase/llm/factory.py) 里加一个分支）：

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
| `RAG_EVAL_LLM_MODEL` / `RAG_EVAL_LLM_BASE_URL` / `RAG_EVAL_LLM_API_KEY` | 沿用上面的回答模型 | 评测裁判（`ragbase eval` 的 Ragas 与中文裁判）单独用另一个模型 / 服务；设置了哪项就覆盖哪项，其余沿用 |

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

- 运行全部单元测试：`uv run python tests/run_all.py`（每个 `tests/test_*.py` 一个子进程，打印各自汇总行，任一失败则非零退出；可带文件名片段只跑部分，如 `run_all.py llm api`）。CI：`.github/workflows/tests.yml`（push / PR 到 main 时在 ubuntu 上 `uv sync --locked`，用 CPU 版 torch 跑同一个脚本）。
- 单元测试（均不加载模型、不碰真实数据，网络只访问本机 127.0.0.1 的桩服务）：`uv run python tests/test_api.py`（假流水线 / 假入库，另有用临时 ChromaStore 的端到端用例）、`tests/test_ingest_service.py`（临时目录 + 假 embedding）、`tests/test_sources.py`（按书过滤、快照刷新、分词缓存）、`tests/test_vectorstores.py`（Chroma / Qdrant 同一套契约测试、迁移、工厂与命令行）、`tests/test_toc.py`（用 fitz 生成带书签的 PDF）、`tests/test_config.py`、`tests/test_llm.py`（本机 OpenAI 兼容桩服务：参数透传、5xx 重试、超时 -> 504、`/ask` 端到端）、`tests/test_parser.py`（临时生成的 PDF / DOCX / GB18030 文本）。
- 依赖：文件上传需要 `python-multipart`（已列入 `pyproject.toml` 的依赖，旧环境请重新执行 `uv sync`）。

## 对话 Agent

在 `/ask`（单轮、固定流程）之外，新增多轮对话 Agent（Stage 4）：由大模型决定是否检索、检索什么，再带 `[n]` 引用作答。`/ask` 不受影响。

```text
用户消息 → Agent（create_agent，LangGraph）─┬→ search_knowledge(query, sources?) → 召回 + 精排 + 拒答 / 上下文过滤（同 /ask）
                                           ├→ list_documents()                  → 书目与片段数
                                           └→ 最终回答（[n] 引用）
会话历史：InMemorySaver，thread_id = session_id
```

- 代码：[src/ragbase/agent/tools.py](src/ragbase/agent/tools.py)（工具与引用登记表）、[src/ragbase/agent/service.py](src/ragbase/agent/service.py)（Agent、会话、限制）、[src/ragbase/cli.py](src/ragbase/cli.py)（命令行 `ragbase chat`）。
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

命令行：`uv run ragbase chat [--source 书名.pdf]`（`/new` 新会话，`exit` 退出）。

局限：
- 会话只存在服务进程内存里，重启即丢；超过 `RAG_CHAT_MAX_SESSIONS` 或空闲超过 `RAG_CHAT_SESSION_TTL` 的会话被淘汰（不持久化、不跨进程）。
- **尚未用真实大模型验证**：单元测试用脚本化的假模型；豆包 / 其他 OpenAI 兼容服务需支持 `tools` / `tool_calls`，这是假设，未验证。
- 流式 `token` 是模型每次生成的文本增量；若模型在调用工具前先说了一句话，这句话也会出现在 token 流里，以 `done.answer` 为准。
- 单元测试：`uv run python tests/test_agent.py`（假检索 + 假模型，不调用真实大模型）。

## 接口鉴权

本产品**没有用户体系**，只有一把静态共享密钥：设置 `RAG_API_KEY` 后，所有 API 请求都必须带上它；不设置（或留空）则不鉴权，行为与之前完全一致，适合本机使用。

```bash
# .env 或环境变量
RAG_API_KEY=换成一串足够长的随机字符串     # 如：openssl rand -hex 32
uv run ragbase serve --host 0.0.0.0        # 对外暴露时务必设置密钥（否则启动日志会警告）
```

请求头两种写法等价（同时带时任一匹配即可）：

```bash
curl -H "Authorization: Bearer $RAG_API_KEY" -X POST localhost:8000/search -H 'Content-Type: application/json' -d '{"query":"挣值分析"}'
curl -H "X-API-Key: $RAG_API_KEY" localhost:8000/documents
curl -N -H "Authorization: Bearer $RAG_API_KEY" -X POST localhost:8000/chat/stream -H 'Content-Type: application/json' -d '{"message":"你好"}'
```

缺少或错误的密钥返回 `401`，响应头带 `WWW-Authenticate: Bearer`，响应体是统一错误格式 `{"error":"unauthorized","detail":"..."}`；`/chat/stream` 在开始流式前就被拒绝，大文件上传在读取请求体前就被拒绝。

**无需密钥的路径**（只限 GET / HEAD）：

| 路径 | 说明 |
| --- | --- |
| `/`、`/library`、`/c/{id}`、`/assets/*` | 前端页面与静态资源（里面没有数据，数据都走受保护的接口） |
| `/docs`、`/redoc`、`/openapi.json` | 交互文档与接口描述（不含密钥） |
| `/health` | 没有有效密钥时只返回 `{"status": "...", "auth_required": true}`（存活探针 / 让前端知道要输入密钥）；带有效密钥返回完整内容；未启用鉴权时返回完整内容和 `"auth_required": false` |

其余一切（`/search`、`/ask`、`/chat*`、`/documents*`、`/ingest*`，以及以后新增的任何路由）默认都需要密钥：鉴权是“默认拒绝 + 显式公开白名单”（[src/ragbase/api/auth.py](src/ragbase/api/auth.py)），不会因为忘了给新路由加依赖而裸奔。命令行（`ragbase ask` / `ingest` 等）在进程内直接运行，不经过 HTTP，不需要密钥。

**Web 界面**：打开页面时若服务要求密钥，会弹出“需要访问密钥”对话框；密钥通过 `/health` 校验后保存在浏览器 localStorage（`ragbase.apiKey`），之后每个请求（含流式对话和上传）自动带上。任何请求返回 401（如密钥被更换）都会清掉本地密钥并重新弹出对话框，输入框里未发送的文字不受影响；右上角状态指示的弹层里有“更换密钥”。

**CORS**：默认不加 CORS 中间件（前端与 API 同源托管，不需要）。前端部署在别的域名时设置 `RAG_CORS_ORIGINS=https://app.example.com,http://localhost:5173`（逗号分隔），服务会允许这些来源携带 `Authorization` / `X-API-Key` / `Content-Type` 头访问，并暴露 `X-Request-ID`；预检请求不需要密钥。

安全说明：
- 密钥是明文共享的 Bearer 令牌：对外暴露时务必放在 HTTPS 反向代理（nginx / Caddy 等）后面，否则密钥会在网络上明文传输。
- 比较使用常量时间算法；服务不会把密钥或请求头里的值写进日志或错误响应。注意反向代理自己的访问日志不要记录 `Authorization` 头。
- 密钥保存在浏览器 localStorage，同源页面里的脚本可读取——不要在同一域名下托管不可信的页面。
- 轮换密钥：修改环境变量 / `.env` 里的 `RAG_API_KEY` 后重启服务（会话在内存里，重启会丢失）；旧密钥立即失效，浏览器端下次请求收到 401 后会提示重新输入。
- 没有限流、没有按人区分的审计：它只解决“别让没拿到密钥的人用你的服务”，不是完整的用户权限系统。
- 单元测试：`uv run python tests/test_auth.py`。

## Web 界面

`web/` 是前端工程（React + Vite + TypeScript + Tailwind CSS + shadcn/ui；路由 react-router，数据请求 TanStack Query，回答用 react-markdown 渲染），Stage 5 新增。需要 Node 22 与 pnpm。

```bash
# 开发：后端在 8000 端口运行（uv run ragbase serve），前端 dev server 把 API 请求代理过去
cd web && pnpm install && pnpm dev          # http://localhost:5173，热更新

# 生产：构建后由 FastAPI 在同一端口托管，不需要另起前端服务
cd web && pnpm install && pnpm build        # 产物在 web/dist
uv run ragbase serve                        # 浏览器打开 http://127.0.0.1:8000/
```

- **托管方式**：`web/dist` 存在时，API 服务提供 `/assets/*`（带内容哈希，长缓存）和前端路由 `/`、`/library`、`/c/{session_id}`（返回 `index.html`）；这些路径不会遮住 `GET /documents`、`/chat/sessions`、`/health` 等 JSON 接口。目录是否存在在请求时检查，服务运行中重新 `pnpm build` 立即生效；未构建时 `GET /` 返回提示“前端未构建：cd web && pnpm install && pnpm build”。构建目录可用 `RAG_WEB_DIST` 改（相对路径按项目主目录解析）。
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

- 切分大小与重叠：[src/ragbase/ingest/splitter.py](src/ragbase/ingest/splitter.py)
- 召回数、精排数、阈值、设备、服务地址、前端构建目录：[src/ragbase/config.py](src/ragbase/config.py)（均可用 `RAG_*` 环境变量覆盖，见“服务化运行”）；`ragbase ask` / `ragbase eval` 的命令行参数优先级更高
- 本地模型与设备：[src/ragbase/models/embeddings.py](src/ragbase/models/embeddings.py)、[src/ragbase/models/reranker.py](src/ragbase/models/reranker.py)

扫描版 PDF 或文本层质量较差的 PDF 需要另行增加 OCR；当前项目不会自动 OCR。

## 常见问题

- `AuthenticationError`：检查 `.env` 中的 `RAG_LLM_API_KEY`、`RAG_LLM_MODEL` 和 `RAG_LLM_BASE_URL`（旧的 `ARK_*` 同样有效）。
- 找不到向量库：先运行 `uv run ragbase ingest`（或 `--full` 全量重建）；也请确认当前目录（或 `RAG_HOME`）是项目主目录。
- MPS 不可用：设置环境变量 `RAG_DEVICE=cpu`（默认 `auto` 会自动回落到 CUDA / CPU）。
- 依赖环境异常：重新执行 `uv sync`。
- 不知道哪里没配好：运行 `ragbase doctor`，按 `✗` / `!` 项的提示修复。
- `ragbase serve` 提示“本地模型尚未下载”：运行 `ragbase models download`（国内网络加 `HF_ENDPOINT=https://hf-mirror.com`）。
- 模型下载很慢 / 超时：见“快速开始 -> 国内网络”；也可以在能访问 Hugging Face 的机器上下载后，把整个 `HF_HOME` 目录（或 Docker 的 `ragbase-models` 卷）拷过来。
