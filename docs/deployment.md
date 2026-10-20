# 部署

三种方式最终效果一样（浏览器打开 <http://127.0.0.1:8000/>）：Docker Compose、`scripts/start.sh`、手动 `uv`。再加上反向代理 / HTTPS、安全清单、备份与升级。

> **状态说明**：Docker 相关文件（`Dockerfile`、`docker-compose.yml`）**还没有在真实环境里完成过镜像构建和运行验证**，只做过静态校验。本文里 Docker 部分的命令按文件内容推导，请以你本机的实际结果为准。反向代理片段同样是示例，没有在真实 nginx / Caddy 上跑过。

## 目录

- [Docker Compose](#docker-compose)
- [scripts/start.sh](#scriptsstartsh)
- [手动安装](#手动安装)
- [反向代理与 HTTPS](#反向代理与-https)
- [安全清单](#安全清单)
- [备份](#备份)
- [升级](#升级)

## Docker Compose

CPU 版，不用装 Python / Node。镜像是多阶段构建：Node 22 构建 Web 界面 → `python:3.12-slim` 里用 uv 按 `uv.lock` 安装依赖（Linux 上 torch 走 PyTorch 官方 CPU 源）→ 最小运行镜像（非 root 用户 uid 1000、带 `libgomp1`、`ENTRYPOINT ["ragbase"]`、默认 `CMD serve --host 0.0.0.0 --port 8000`）。镜像**不含模型权重**。

### 首次运行

```bash
cp .env.example .env              # 编辑 .env：填 RAG_LLM_*；对外开放服务必须设置 RAG_API_KEY（见安全清单）
mkdir -p ragbase-home/data        # 项目主目录（挂载为容器内 /data）；把文档放进 ragbase-home/data/
docker compose run --rm ragbase models download   # 首次：下载模型到 ragbase-models 卷（约 2.2 GB，之后复用）
docker compose run --rm ragbase ingest            # 建立索引
docker compose up -d --build                      # 启动，浏览器打开 http://127.0.0.1:8000/
docker compose run --rm ragbase doctor            # 随时体检
```

- **国内网络**下载模型慢或不通：`HF_ENDPOINT=https://hf-mirror.com docker compose run --rm ragbase models download`；也可以把 `HF_ENDPOINT=https://hf-mirror.com` 写进 `.env`，compose 会自动读取并传给容器。
- 第 3 步可以省略：`docker compose up` 的启动命令带 `--download-models`，模型缺失时会自动下载（但下载期间服务还没起来）。
- 全云端的 embedding / reranker 配置（见 [models.md](models.md)）不需要下载模型，第 3 步会直接提示“无需下载”。
- 首次启动时索引仍为空，服务可以启动但不能问答；先完成第 4 步（或在 Web 界面上传文档，见下）。

### 卷与目录布局

| 位置 | 容器内 | 说明 |
| --- | --- | --- |
| `./ragbase-home`（绑定目录） | `/data`（= `RAG_HOME`） | 项目主目录：文档、索引、缓存、入库记录、数据库（会话 / 题集 / 体检报告）都在这里 |
| `ragbase-models`（命名卷） | `/models`（= `HF_HOME`） | Hugging Face 模型缓存 |
| `qdrant-storage`（命名卷，仅 qdrant profile） | `/qdrant/storage`（在 qdrant 容器里） | Qdrant 服务端的数据 |

容器内 `RAG_HOME=/data`，所以布局是（文档在 `/data/data/`，看起来别扭但与本机布局完全一致）：

```text
/data/                    ← 宿主机 ./ragbase-home
  data/                   你的文档（宿主机 ragbase-home/data/）
  chroma_db/              向量索引（chroma）
  qdrant_db/              向量索引（嵌入式 qdrant 时）
  ragbase.db              内置 SQLite 数据库：对话会话、评测题集、体检报告（运行时旁边还有 -wal / -shm）
  .ingest_meta.json       增量入库的文件指纹
  .index_info.json        索引的 embedding 签名
  .bm25_cache/            BM25 分词缓存
  .env                    （可选）也会被读取，见下
/models/                  Hugging Face 缓存
```

配置的两种来源：根目录 `.env` 通过 compose 的 `env_file` 注入容器环境；`/data/.env`（即 `ragbase-home/.env`）因为 `RAG_HOME=/data` 也会被应用读取。两者**二选一即可**；同名变量以容器环境（`env_file`）为准。**不要在给 compose 用的 `.env` 里设置 `RAG_DATA_DIR` / `RAG_DB_DIR` 等路径变量**，宿主机路径在容器里不存在。

镜像里已设置 `RAG_HOME=/data`、`HF_HOME=/models`、`RAG_DEVICE=cpu`、`CHROMA_TELEMETRY_DISABLED=true`。`HEALTHCHECK` 每 30 秒请求容器内的 `/health`（启动宽限 120 秒；启用鉴权时精简版 `/health` 也返回 200）。

### 端口与监听地址

默认只绑定宿主机的 `127.0.0.1:8000`（容器内服务监听 `0.0.0.0:8000`）。要让局域网访问，设置 `RAGBASE_BIND=0.0.0.0`（可选 `RAGBASE_PORT`，在 shell 或 `.env` 里）——**务必同时设置 `RAG_API_KEY`**，并放到 HTTPS 反向代理后面。

### 日常使用

- **新增文档**：服务在线时到 Web 界面“知识库”页上传（在线增量入库；前提是启动时索引非空）；或者
  `docker compose stop ragbase` → 把文件放进 `ragbase-home/data/` → `docker compose run --rm ragbase ingest` → `docker compose up -d`。
- **`ingest` 不要与运行中的服务同时跑**；`ingest --full` 更要停服务（见下面的警告）。
- **Linux 上** `ragbase-home` 要由你自己先 `mkdir -p ragbase-home/data`（容器里是 uid 1000 的非 root 用户，让 Docker 以 root 创建的目录它写不进去）；不想管权限可以在 `docker-compose.yml` 里改成命名卷 `ragbase-home:/data`（并在底部 `volumes:` 里声明）。

### 可选：Qdrant 服务端做向量库

默认用 Chroma，不需要它。在 `.env` 里加：

```dotenv
RAG_VECTOR_STORE=qdrant
RAG_QDRANT_URL=http://qdrant:6333
```

然后 `docker compose --profile qdrant up -d` 同时启动两个服务（`qdrant/qdrant:latest` 镜像，未固定版本）。已有 Chroma 索引想迁移，不重新向量化：

```bash
docker compose stop ragbase
docker compose --profile qdrant up -d qdrant             # 迁移需要 Qdrant 服务在运行
# .env 里先只加 RAG_QDRANT_URL；确认迁移成功后再把 RAG_VECTOR_STORE 改为 qdrant
docker compose run --rm ragbase migrate-store --to qdrant
docker compose up -d
```

（迁移命令的行为见 [architecture.md](architecture.md#向量库抽象)；上述 compose 编排未经实测。）

## scripts/start.sh

克隆仓库后一键（macOS / Linux）：

```bash
git clone <本仓库地址> ragbase && cd ragbase
scripts/start.sh                  # 国内网络：HF_ENDPOINT=https://hf-mirror.com scripts/start.sh
```

脚本依次：检查 `uv`（没有会打印安装命令）→ `uv sync --locked` → 构建 Web 界面（`web/dist` 不存在且装了 `pnpm` 时）→ 没有 `.env` 就运行 `ragbase init`（已有则只补齐缺失的 `data/` 等）→ `ragbase models download` → `ragbase doctor`（有 `✗` 就停）→ `ragbase serve`。可重复运行，已完成的步骤会跳过。

- 额外参数会传给 `ragbase serve`：`scripts/start.sh --host 0.0.0.0 --port 8080`；`--no-serve` 只做准备、不启动。
- 项目主目录：`RAG_HOME`，未设置则为仓库根目录。
- 首次运行后请编辑 `.env` 填入大模型配置，把文档放进 `data/`，**另开终端**运行 `uv run ragbase ingest`；服务在索引为空时启动，建好索引后需要重启服务。更省事的顺序是：`scripts/start.sh --no-serve` → 编辑 `.env`、放文档 → `uv run ragbase ingest` → `uv run ragbase serve`。

## 手动安装

需要 Python ≥ 3.12 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync                           # 按 uv.lock 创建 .venv 并以可编辑模式安装 ragbase
uv run ragbase init               # 生成 data/、.env、bm25_userdict.txt（不会覆盖已有文件）
$EDITOR .env                      # 填 RAG_LLM_BASE_URL / RAG_LLM_API_KEY / RAG_LLM_MODEL
uv run ragbase models download    # 下载 embedding + reranker（已缓存则跳过）
cp ~/你的资料/*.pdf data/          # 放文档
uv run ragbase ingest             # 建立索引（首次会向量化全部文档，耗时与资料量有关）
uv run ragbase doctor             # 体检
uv run ragbase serve              # http://127.0.0.1:8000/
```

- Web 界面需要前端构建产物：仓库里运行 `scripts/build_web.sh`（Node 22 + pnpm；产物在 `web/dist`，同时复制一份到 `src/ragbase/web/dist` 以便打进 wheel）。没有构建时 API 仍可用，页面会提示“前端未构建”。
- 在别的目录运行时，先设置环境变量 `RAG_HOME` 指向项目主目录（见 [configuration.md](configuration.md#路径与项目主目录)）。
- 长期运行可交给 systemd / launchd / supervisor 等。一个 systemd 单元的**示例**（路径、用户自行替换，未经验证）：

  ```ini
  [Service]
  WorkingDirectory=/opt/ragbase
  Environment=RAG_HOME=/opt/ragbase
  ExecStart=/opt/ragbase/.venv/bin/ragbase serve --host 127.0.0.1 --port 8000
  Restart=on-failure
  ```

> **不要同时运行的操作**：
> - 服务运行期间不要运行 `ragbase ingest --full`：它会清空整个向量库（删除集合并重建），而服务进程还持有该索引。服务在线时请用 `POST /documents` / `POST /ingest` / `DELETE /documents/{name}`，或先停服务再跑命令行。
> - `ragbase ingest`（增量）同样不要与服务同时运行（两个进程会同时写同一个库）。
> - 嵌入式 Qdrant（只设了 `RAG_VECTOR_STORE=qdrant`、没设 `RAG_QDRANT_URL`）同一个目录同一时刻只允许一个进程打开：`serve` 运行时再运行 `ingest` / `ask` / `migrate-store` 会报 “already accessed by another instance”。要多进程访问请用 Qdrant 服务端模式。
> - 迁移向量库（`migrate-store`）也要先停服务。

## 反向代理与 HTTPS

密钥是明文共享的 Bearer 令牌：**对外暴露时务必放在 HTTPS 反向代理后面**，否则密钥会在网络上明文传输。服务本身只做 HTTP，建议保持监听 `127.0.0.1`，由代理对外。

nginx 示例（**未在真实环境验证**；域名、证书路径自行替换）：

```nginx
server {
    listen 443 ssl;
    server_name rag.example.com;
    ssl_certificate     /etc/ssl/rag.example.com/fullchain.pem;
    ssl_certificate_key /etc/ssl/rag.example.com/privkey.pem;

    client_max_body_size 300m;          # 不小于 RAG_MAX_UPLOAD_MB（默认 300），否则上传在代理层就被 413 拒绝

    # SSE 流式对话：必须关闭缓冲，否则 token 会攒到最后才一次性到达
    location = /chat/stream {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;         # 大模型慢时别被代理超时掐断
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;         # /ask 与入库后的 BM25 刷新可能较慢
    }
}

server {
    listen 80;
    server_name rag.example.com;
    return 301 https://$host$request_uri;
}
```

- 服务对 `/chat/stream` 已经发送 `X-Accel-Buffering: no`，nginx 会据此关闭缓冲；显式写 `proxy_buffering off` 是双保险。其他代理（Caddy、Traefik、云厂商负载均衡）也要确认对 `text/event-stream` 不缓冲、不压缩、超时足够长。
- 前端与 API 同源托管，通过代理访问时**不需要** CORS 配置。
- 不要让代理的访问日志记录 `Authorization` / `X-API-Key` 请求头（nginx 默认的 `combined` 格式不记录请求头，自定义 `log_format` 时别加 `$http_authorization`）。

## 安全清单

本产品**没有用户体系**，只有一把静态共享密钥；它只解决“别让没拿到密钥的人用你的服务”。逐条检查：

- [ ] **设置了 `RAG_API_KEY`**（只要不是纯本机使用：`RAG_HOST` 不是回环地址、Docker 发布到非 `127.0.0.1`、放到公网）。否则任何能访问端口的人都能查询你的知识库、消耗你的大模型额度、上传 / 删除文档。服务以非回环地址监听且没有密钥时，启动日志会警告；`ragbase doctor` 也会警告。
  生成密钥：`ragbase init --api-key`（只在**新建** `.env` 时写入；已有 `.env` 会提示，要重新生成加 `--force`，旧 `.env` 备份为 `.env.bak`）或 `openssl rand -hex 32`。
- [ ] **放在 HTTPS 反向代理后面**（见上）。
- [ ] **CORS**：默认不加。只有前端部署在别的域名时才设置 `RAG_CORS_ORIGINS`，且只列确切来源。
- [ ] **代理 / 日志**：代理访问日志不记录 `Authorization` 与 `X-API-Key`。服务自己不会把密钥或请求头里的值写进日志或错误响应。
- [ ] **浏览器端**：Web 界面把密钥保存在浏览器 localStorage（`ragbase.apiKey`），同源页面里的脚本可读取——不要在同一域名下托管不可信的页面。
- [ ] **`.env` 保护**：含密钥，已被 `.gitignore` 与 `.dockerignore` 排除（不会进仓库和镜像）；`ragbase init` 生成时权限为 `0600`。
- [ ] **上传上限**：`RAG_MAX_UPLOAD_MB` 与代理的 `client_max_body_size` 一致。
- [ ] **没有限流、没有按人区分的审计**：需要的话在代理层做限流。
- [ ] **轮换密钥**：修改环境变量 / `.env` 里的 `RAG_API_KEY` 后重启服务（会话保存在数据库里，重启不丢）；旧密钥立即失效，浏览器端下次请求收到 401 后会提示重新输入。
- [ ] Docker：默认只绑定 `127.0.0.1`；容器以非 root 用户运行。

## 备份

| 内容 | 位置（`RAG_HOME` 下） | 是否必须备份 |
| --- | --- | --- |
| 原始文档 | `data/` | **是**——它是一切的来源，索引可以从它重建 |
| 配置与密钥 | `.env` | **是**（含密钥，注意保管） |
| 数据库 | `ragbase.db`（`RAG_DB_PATH` 可改位置） | **是**——对话会话（含引用）、评测题集、体检报告历史都只存在这里，**无法从别处重建**，见下 |
| 领域词典 | `bm25_userdict.txt` | 建议 |
| 向量索引 | `chroma_db/`（或 `qdrant_db/` / Qdrant 服务端的卷） | 可选：不备份就用 `ragbase ingest --full` 重建（要重新向量化，耗时 / 云端会产生费用） |
| 入库记录与签名 | `.ingest_meta.json`、`.index_info.json` | 与向量索引**一起**备份、一起恢复 |
| BM25 分词缓存 | `.bm25_cache/` | 否，启动时会按需重建 |
| 模型缓存 | `HF_HOME`（Docker：`ragbase-models` 卷） | 否，可重新下载 |

仓库根目录的 `eval_questions.json` 只在首次启动时被导入一次，之后题集以数据库为准；想保留一份文本形式的题集，用 `ragbase questions export FILE` 导出。

### 备份数据库（`ragbase.db`）

数据库开启了 WAL 模式：写入先进入 `ragbase.db-wal`，之后才合并回 `ragbase.db`，`ragbase.db-shm` 是配套的共享内存索引文件。因此服务**运行中**只拷贝 `ragbase.db` 这一个文件，可能丢掉还留在 `-wal` 里的最近写入，甚至拷到不一致的状态。二选一：

1. **停服务后拷贝**（最简单）：`ragbase serve` 正常停止后，`ragbase.db` 已经是完整的，拷走即可。如果旁边还留着 `ragbase.db-wal`（进程被强杀、断电之后会这样），要把 `ragbase.db`、`ragbase.db-wal`、`ragbase.db-shm` **三个文件一起拷**，不要只拷其中一个。
2. **在线备份**：用 SQLite 自带的备份命令，服务不用停，得到的是一致的快照文件（不需要再拷 `-wal` / `-shm`）：

   ```bash
   sqlite3 ragbase.db ".backup ragbase-backup.db"
   # 没有 sqlite3 命令行时，用 Python 标准库等价（在 ragbase 的环境里运行，路径自行替换）：
   uv run python -c "import sqlite3; s=sqlite3.connect('ragbase.db'); d=sqlite3.connect('ragbase-backup.db'); s.backup(d); d.close(); s.close()"
   ```

恢复：停服务，把备份文件放回 `RAG_DB_PATH`（或 `RAG_HOME/ragbase.db`）并**删掉旧的 `-wal` / `-shm` 文件**（否则旧的 `-wal` 可能被套用到新文件上），再启动。

**Docker Desktop（macOS）绑定目录上的已知风险**：SQLite 的 WAL 模式依赖同一台主机上的共享内存和文件锁，官方文档明确说它不适用于网络文件系统。Docker Desktop 在 macOS 上把宿主机目录经由虚拟化的文件共享层挂进容器，默认的绑定目录 `./ragbase-home` 就是这种情况，可能出现 `database is locked`，极端情况下数据库损坏的风险也更高。其他环境（命名卷、Linux 的绑定目录、本机直接运行）按理没有这个问题，但同样没有逐一实测。以上是按 SQLite 的已知限制做的推断，**本项目的 Docker 部署本身尚未验证，没有在 macOS 的 Docker Desktop 上实测过**；在 macOS 上用 Docker 时，建议把 `docker-compose.yml` 里的 `./ragbase-home` 改成命名卷 `ragbase-home:/data`（并在底部 `volumes:` 里声明；做法见上文“Linux 上……命名卷”一条），并勤做备份。

备份向量索引前最好先停服务，以免拷到写到一半的 SQLite 文件（这是通用的稳妥做法，未针对 Chroma / Qdrant 做专门验证）。恢复后用 `ragbase doctor` 检查。

## 升级

项目目前**未发布**（`0.1.0`），升级前先看 [CHANGELOG.md](../CHANGELOG.md)。

- **源码部署**：`git pull` → `uv sync --locked` → 需要界面更新时 `scripts/build_web.sh` → 重启服务。
- **Docker**：`git pull` → `docker compose up -d --build`。数据在 `ragbase-home` 与 `ragbase-models`，不随镜像重建丢失。
- **索引是否需要重建**：
  - embedding 配置没变 → 索引仍可用（签名比对通过）。换了 embedding 提供方 / 模型 / 维度 → 必须 `ragbase ingest --full`，见 [models.md](models.md#换-embedding索引签名规则)。
  - 解析 / 清洗 / 切分规则的版本号（代码里的 `INGEST_PIPELINE_VERSION`）变化时，下次增量入库会把所有文件视为已变化并重新索引——旧索引在此之前照常可用，只是缺少新规则带来的信息（例如版本 3 起 PDF 片段带章节标题）。
  - BM25 分词规则版本变化时，分词缓存自动失效并在启动时重建，无需操作。
- **数据库**（`ragbase.db`）：表结构有版本号，升级后第一次启动 `ragbase serve`（或运行用到数据库的命令）时**自动迁移**到最新，不需要手动操作；`ragbase doctor` 会提示“需要升级”。迁移不可逆，**升级前先备份 `ragbase.db`**。不支持降级：用旧版本程序打开被新版本升级过的数据库，会因为版本过新而拒绝启动并提示升级程序。
- 从还把会话放在内存里的旧版本升级：之前的内存会话本来就随重启丢失，没有可迁移的内容；`RAG_CHAT_MAX_SESSIONS` 和 `RAG_CHAT_SESSION_TTL` 已废弃，`.env` 里还留着时启动日志会警告，删掉即可。
- 换 reranker 或大模型：不需要重建索引，但换 reranker 要[重新标定阈值](models.md#reranker-换模型后重新标定拒答阈值)。
