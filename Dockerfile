# syntax=docker/dockerfile:1
# ragbase（CPU 版）：多阶段构建。
#   docker build -t ragbase .
#   docker run --rm -p 8000:8000 -v ragbase-data:/data -v ragbase-models:/models -e RAG_API_KEY=... ragbase
# 一般直接用 docker compose（见 docker-compose.yml）。镜像不含模型权重：首次启动前运行
#   docker compose run --rm ragbase models download      （国内网络加 HF_ENDPOINT=https://hf-mirror.com）

# ---- 阶段 1：构建 Web 界面 ------------------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
RUN corepack enable
COPY web/package.json web/pnpm-lock.yaml ./
RUN --mount=type=cache,id=pnpm-store,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile
COPY web/ ./
RUN pnpm build

# ---- 阶段 2：安装 Python 依赖与 ragbase（Linux 上 torch 自动走 CPU 源，见 pyproject.toml 的 tool.uv.sources）----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
# 先只装第三方依赖：依赖不变时这一层（数 GB）命中构建缓存
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
# 再装项目本身（非可编辑），并把前端产物放进包目录，hatchling 会把它打进 wheel（pyproject.toml 的 artifacts）
COPY src ./src
COPY --from=web /web/dist ./src/ragbase/web/dist
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ---- 阶段 3：运行镜像 ----------------------------------------------------------------------
FROM python:3.12-slim
# libgomp1：onnxruntime（chromadb 依赖）在 slim 镜像上需要
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 ragbase \
    && mkdir -p /data /models \
    && chown ragbase:ragbase /data /models
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    RAG_HOME=/data \
    HF_HOME=/models \
    RAG_DEVICE=cpu \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    CHROMA_TELEMETRY_DISABLED=true
USER ragbase
WORKDIR /data
# /data = 项目主目录（RAG_HOME）：data/ 文档、chroma_db/ 索引、.env 等；/models = Hugging Face 模型缓存
VOLUME ["/data", "/models"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
ENTRYPOINT ["ragbase"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
