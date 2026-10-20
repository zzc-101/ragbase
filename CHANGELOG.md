# 更新日志

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/)。项目尚未发布，下面的 0.1.0 汇总了目前为止的全部能力（来自 git 提交历史）。

## 0.1.0（未发布）

### 检索与问答

- **混合检索**：向量召回 + BM25（`jieba` + `bm25s`）两路，用 RRF 融合；`--mode vector|bm25|hybrid` 可切换，API 可按请求切换。
- **精排**：对候选逐个打 P(相关) 分（0 ~ 1），取 Top 5。
- **低分拒答**：最相关片段得分低于阈值（默认 0.7）直接拒答，**不调用大模型**；只把得分足够高的片段（默认 ≥ 0.95，top1 必留）交给大模型。
- **引用溯源**：提示词要求先逐字摘录【依据】再作答，解析后分别返回；回答里的 `[片段n]` / `[n]` 对应到带文件名、页码、章节路径、各路得分的片段。
- **章节标题**：入库时读取 PDF 书签，给片段加上 `heading_path`（如 `第11章 … > 11.3 …`），参与向量化和 BM25，不参与精排。
- **按书过滤**：`/search`、`/ask`、对话和 `ragbase ask --source` 支持只在指定文件里检索。
- **增量入库**：文件指纹（版本 + 文件名 + 大小 + 修改时间）；新增 / 修改 / 删除自动识别，逐文件落盘记录，失败的文件下次重试；旧版索引自动迁移。
- 修复早期问题：裁判把“不正确”误判为“正确”、空召回仍调用大模型、索引缺失时增量入库静默跳过、`.doc` / 大写扩展名 / `README` 被误入库、非 PDF 来源页码错误。

### 对话 Agent

- `create_agent`（LangChain + LangGraph）+ 工具 `search_knowledge`（复用召回 / 精排 / 拒答 / 过滤）与 `list_documents`；会话记忆持久化到内置 SQLite（见下）。
- 每轮工具调用 / 模型调用次数上限；发给大模型的历史按整轮截取最近 N 轮；同会话请求串行；失败 / 断开的轮次整轮回滚。
- **会话级引用编号**：同一片段在整个会话内编号不变，历史消息带引用。
- 接口：`POST /chat`、`POST /chat/stream`（SSE）、`GET /chat/sessions`、`GET|DELETE /chat/sessions/{id}`；命令行 `ragbase chat`。
- 局限：未用真实大模型验证工具调用兼容性；系统提示词里写着作者语料的领域文案。

### 内置数据库与会话持久化

- **内置 SQLite**（`ragbase/db/`，只用标准库 `sqlite3` 与 `langgraph-checkpoint-sqlite`）：默认 `RAG_HOME/ragbase.db`，新增配置项 `RAG_DB_PATH` 可覆盖；WAL 模式；表结构有版本号（`PRAGMA user_version`），启动时自动建库 / 迁移；`ragbase doctor` 增加数据库检查。备份方式与 Docker Desktop（macOS）绑定目录上的 WAL 风险见 [docs/deployment.md](docs/deployment.md#备份)。
- **会话持久化**：会话的完整历史、会话列表、会话级引用编号与引用片段保存在数据库里，重启后接着追问，编号不变（片段身份改用稳定的 md5，不再依赖每个进程不同的 `hash()`）；会话只在手动删除时才删除；状态里保留全部消息，只在调用大模型时截取最近 `RAG_CHAT_HISTORY_TURNS` 轮；进程中途被杀留下的半截轮次在下一轮开始时自动清理；每轮只保留最新的 checkpoint，库不会随轮数膨胀。大模型未配置时，会话列表、历史、删除照常可用。命令行 `ragbase chat` 仍只用内存、不写库。
- **废弃两个会话配置项**：`RAG_CHAT_MAX_SESSIONS`、`RAG_CHAT_SESSION_TTL` 不再生效（会话不再自动淘汰）；旧 `.env` 里留着它们仍能启动，只在启动日志里警告，配置模板里已移除。
- **题集管理**：评测题集存进数据库（新增 `reference` 字段，旧的 `answer_keyword` 仍兼容；`reference` 为空的题 `ragbase eval` 跳过）；首次启动自动导入 `eval_questions.json`（只导入一次）；新增 `ragbase questions import|export`，导入整体校验、失败时逐题报错；`ragbase eval` 改为从数据库读题。
- **体检报告历史**：`ragbase diagnose` 的结果存进数据库并保留历史（来源标记 `cli` / `web`），不再默认写 `diagnose_report.json`，需要时用 `--json PATH` 导出；报告额外记录所用题目的 id 与内容摘要、体检期间的对话请求数。
- **Web“体检”页**：在 Web 上发起体检（后台运行、显示进度）、查看报告与历史、删除报告、两份报告对比（题集不同、来源不同、体检期间有对话请求时给出提示）、题集的增删改与导入导出。
- **新增接口**：`/eval-questions`（增删改查、`/import`、`/export`）、`POST /diagnose`、`GET /diagnose/jobs/{id}`、`/diagnose/reports*`；体检与入库 / 删除文档共用一把“忙”锁，冲突返回 409；新增前端路由 `/diagnose`。见 [docs/api.md](docs/api.md)。

### 服务与界面

- **HTTP 服务**（FastAPI）：`/health`、`/search`、`/ask`、`/chat*`、`/documents*`（上传 / 列表 / 删除）、`/ingest*`、`/eval-questions*`、`/diagnose*`；统一错误格式、`X-Request-ID`、访问日志；大模型超时 504、其他失败 502；单进程。
- **Web 界面**（React + Vite + shadcn/ui）：对话页（会话列表、流式回答、引用角标与“依据”面板、工具调用步骤、按书限定范围）、知识库页（拖拽上传、入库进度、书目卡片、删除）、体检页、状态指示、深色模式、响应式布局。
- **静态 API Key 鉴权**：`RAG_API_KEY`；“默认拒绝 + 公开白名单”，在读取请求体前拒绝；`/health` 无密钥时只返回精简内容；`RAG_CORS_ORIGINS`；对外暴露且未设密钥时启动告警。

### 模型与向量库

- **embedding 与 reranker 可选本地或云端**：本地 Qwen3-Embedding-0.6B / Qwen3-Reranker-0.6B；云端 OpenAI 兼容 `/embeddings`、通用 `/rerank`；reranker 可设为 `none`。地址 / 密钥未配置时回退到大模型配置（仅同源时沿用密钥）。全云端时不加载 torch。
- **索引签名守卫**：`.index_info.json` 记录建索引时的 embedding 提供方 / 模型 / 维度，配置不一致时拒绝并提示 `ragbase ingest --full`。
- **大模型走 OpenAI 兼容协议**：`RAG_LLM_*`（兼容旧的 `ARK_*`）；超时与自动重试；评测裁判可用 `RAG_EVAL_LLM_*` 单独指定。
- **向量库可插拔**：Chroma（默认）与 Qdrant（本地嵌入式或服务端）；`ragbase migrate-store` 在后端之间迁移，不重新向量化。
- 设备自动选择 mps > cuda > cpu。

### 开箱即用

- 命令行 `ragbase`：`init`、`models download`、`doctor`、`serve`、`ingest [--full]`、`ask`、`chat`、`eval`、`diagnose`、`questions import|export`、`migrate-store`。
- `ragbase init`（生成 `data/`、`.env` 模板、默认词典，可 `--api-key`）、`ragbase doctor`（环境自检，`--check-llm` / `--check-models`）、`ragbase models download`（支持 `HF_ENDPOINT` 镜像）、`ragbase serve --download-models`。
- 一键脚本 `scripts/start.sh`、`scripts/build_web.sh`；`Dockerfile`（CPU）与 `docker-compose.yml`（可选 Qdrant profile）——**Docker 文件尚未在真实环境验证镜像构建和运行**。
- 项目主目录 `RAG_HOME`（默认当前目录）；src 布局的 `ragbase` 包，`pyproject.toml` + `uv.lock` 管理依赖，Linux 用 PyTorch CPU 源。

### 评测

- `ragbase diagnose`：参数与速度体检——复用评测题集，预热后对每题分阶段计时（向量化 / 向量库 / BM25 / 融合 / 精排）、算三组词面检索指标，按固定规则给出调参建议；只测量和建议，不改配置、不调用大模型；报告存进数据库（命令行与 Web 共用同一份历史）。见 [docs/diagnostics.md](docs/diagnostics.md)。
- 词面检索指标挪到 `evaluation/lexical.py`（不依赖 ragas），`evaluation.metrics` 照旧可导入。
- `ragbase eval`：Ragas 五项指标、无需大模型的词面代理指标（要点召回、上下文精确率、MRR，分召回 Top20 / 精排 Top5 / 实际交给大模型的片段三组）、中文裁判；`--retrieval-only` 不调用任何大模型；结果按召回模式分文件。
- `scripts/calibrate_rerank_threshold.py`：拒答阈值校准脚本（不调用大模型）。

### 工程

- 断言脚本式单元测试（`tests/run_all.py`），全部使用假模型 / 桩服务 / 临时目录；GitHub Actions：后端测试（Ubuntu + CPU 版 torch）与前端构建 + lint。
- 文档：`README.md` 与 `docs/`（配置、模型、API、部署、架构、评测、开发、排错）；配置项文档由 `scripts/gen_config_docs.py` 从 `Settings` 生成，`tests/test_docs.py` 防止漂移。
