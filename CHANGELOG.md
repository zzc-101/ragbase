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

- `create_agent`（LangChain + LangGraph）+ 工具 `search_knowledge`（复用召回 / 精排 / 拒答 / 过滤）与 `list_documents`；进程内会话记忆。
- 每轮工具调用 / 模型调用次数上限；历史按整轮裁剪；会话数上限 + 空闲 TTL（LRU 淘汰）；同会话请求串行；失败 / 断开的轮次整轮回滚。
- **会话级引用编号**：同一片段在整个会话内编号不变，历史消息带引用。
- 接口：`POST /chat`、`POST /chat/stream`（SSE）、`GET /chat/sessions`、`GET|DELETE /chat/sessions/{id}`；命令行 `ragbase chat`。
- 局限：会话只存在内存里；未用真实大模型验证工具调用兼容性；系统提示词里写着作者语料的领域文案。

### 服务与界面

- **HTTP 服务**（FastAPI）：`/health`、`/search`、`/ask`、`/chat*`、`/documents*`（上传 / 列表 / 删除）、`/ingest*`；统一错误格式、`X-Request-ID`、访问日志；大模型超时 504、其他失败 502；单进程。
- **Web 界面**（React + Vite + shadcn/ui）：对话页（会话列表、流式回答、引用角标与“依据”面板、工具调用步骤、按书限定范围）、知识库页（拖拽上传、入库进度、书目卡片、删除）、状态指示、深色模式、响应式布局。
- **静态 API Key 鉴权**：`RAG_API_KEY`；“默认拒绝 + 公开白名单”，在读取请求体前拒绝；`/health` 无密钥时只返回精简内容；`RAG_CORS_ORIGINS`；对外暴露且未设密钥时启动告警。

### 模型与向量库

- **embedding 与 reranker 可选本地或云端**：本地 Qwen3-Embedding-0.6B / Qwen3-Reranker-0.6B；云端 OpenAI 兼容 `/embeddings`、通用 `/rerank`；reranker 可设为 `none`。地址 / 密钥未配置时回退到大模型配置（仅同源时沿用密钥）。全云端时不加载 torch。
- **索引签名守卫**：`.index_info.json` 记录建索引时的 embedding 提供方 / 模型 / 维度，配置不一致时拒绝并提示 `ragbase ingest --full`。
- **大模型走 OpenAI 兼容协议**：`RAG_LLM_*`（兼容旧的 `ARK_*`）；超时与自动重试；评测裁判可用 `RAG_EVAL_LLM_*` 单独指定。
- **向量库可插拔**：Chroma（默认）与 Qdrant（本地嵌入式或服务端）；`ragbase migrate-store` 在后端之间迁移，不重新向量化。
- 设备自动选择 mps > cuda > cpu。

### 开箱即用

- 命令行 `ragbase`：`init`、`models download`、`doctor`、`serve`、`ingest [--full]`、`ask`、`chat`、`eval`、`migrate-store`。
- `ragbase init`（生成 `data/`、`.env` 模板、默认词典，可 `--api-key`）、`ragbase doctor`（环境自检，`--check-llm` / `--check-models`）、`ragbase models download`（支持 `HF_ENDPOINT` 镜像）、`ragbase serve --download-models`。
- 一键脚本 `scripts/start.sh`、`scripts/build_web.sh`；`Dockerfile`（CPU）与 `docker-compose.yml`（可选 Qdrant profile）——**Docker 文件尚未在真实环境验证镜像构建和运行**。
- 项目主目录 `RAG_HOME`（默认当前目录）；src 布局的 `ragbase` 包，`pyproject.toml` + `uv.lock` 管理依赖，Linux 用 PyTorch CPU 源。

### 评测

- `ragbase eval`：Ragas 五项指标、无需大模型的词面代理指标（要点召回、上下文精确率、MRR，分召回 Top20 / 精排 Top5 / 实际交给大模型的片段三组）、中文裁判；`--retrieval-only` 不调用任何大模型；结果按召回模式分文件。
- `scripts/calibrate_rerank_threshold.py`：拒答阈值校准脚本（不调用大模型）。

### 工程

- 断言脚本式单元测试（`tests/run_all.py`），全部使用假模型 / 桩服务 / 临时目录；GitHub Actions：后端测试（Ubuntu + CPU 版 torch）与前端构建 + lint。
- 文档：`README.md` 与 `docs/`（配置、模型、API、部署、架构、评测、开发、排错）；配置项文档由 `scripts/gen_config_docs.py` 从 `Settings` 生成，`tests/test_docs.py` 防止漂移。
