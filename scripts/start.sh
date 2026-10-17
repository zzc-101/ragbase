#!/usr/bin/env bash
# 一键启动（macOS / Linux，在克隆下来的仓库里运行）：
#   scripts/start.sh [--no-serve] [ragbase serve 的参数，如 --host 0.0.0.0 --port 8000]
# 依次：检查 uv -> 安装依赖 -> 构建 Web 界面 -> 初始化 .env -> 下载模型 -> 体检 -> 启动服务。
# 可重复运行：已完成的步骤会被跳过。--no-serve 只做准备工作，不启动服务。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  BLUE=$'\033[34m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'
else
  BLUE=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi
step() { printf '\n%s==> %s%s\n' "$BLUE" "$*" "$RESET"; }
ok()   { printf '%s✓ %s%s\n' "$GREEN" "$*" "$RESET"; }
warn() { printf '%s! %s%s\n' "$YELLOW" "$*" "$RESET" >&2; }
die()  { printf '%s✗ %s%s\n' "$RED" "$*" "$RESET" >&2; exit 1; }

SERVE=1
SERVE_ARGS=()
for arg in "$@"; do
  if [ "$arg" = "--no-serve" ]; then SERVE=0; else SERVE_ARGS+=("$arg"); fi
done

# 项目主目录：RAG_HOME，未设置则为仓库根目录（与直接在仓库里运行 ragbase 一致）
export RAG_HOME="${RAG_HOME:-$ROOT}"
RAGBASE=(uv run --locked ragbase)

step "1/7 检查 uv"
if ! command -v uv >/dev/null 2>&1; then
  die "没有找到 uv。请先安装：curl -LsSf https://astral.sh/uv/install.sh | sh （macOS 也可 brew install uv），然后重新打开终端再运行本脚本"
fi
ok "$(uv --version)"

step "2/7 安装 Python 依赖（uv sync --locked）"
uv sync --locked
ok "依赖就绪（需要 Python >= 3.12，本机没有时 uv 会自动下载）"

step "3/7 Web 界面"
if [ -f "$ROOT/web/dist/index.html" ] || [ -f "$ROOT/src/ragbase/web/dist/index.html" ]; then
  ok "前端已构建，跳过"
elif command -v pnpm >/dev/null 2>&1; then
  "$ROOT/scripts/build_web.sh" || warn "前端构建失败，服务仍可启动（只有 API，没有 Web 界面）"
else
  warn "没有 web/dist，且未安装 pnpm：服务只有 API，没有 Web 界面。需要界面请安装 Node 22 + pnpm（corepack enable）后重新运行本脚本"
fi

step "4/7 初始化配置（.env、data/）"
if [ -f "$RAG_HOME/.env" ]; then
  ok "已有 $RAG_HOME/.env，保留不动"
  "${RAGBASE[@]}" init "$RAG_HOME" >/dev/null   # 只补齐缺失的 data/ 等文件，已有文件不会被改
else
  "${RAGBASE[@]}" init "$RAG_HOME"
  warn "已生成 .env 模板：请编辑 $RAG_HOME/.env 填入大模型配置（RAG_LLM_BASE_URL / RAG_LLM_API_KEY / RAG_LLM_MODEL）；对外开放服务前还要设置 RAG_API_KEY"
fi

step "5/7 下载本地模型（已缓存的会跳过；国内网络可先 export HF_ENDPOINT=https://hf-mirror.com）"
"${RAGBASE[@]}" models download || die "模型下载失败。检查网络，或设置镜像后重试：HF_ENDPOINT=https://hf-mirror.com scripts/start.sh"

step "6/7 环境体检（警告不阻塞）"
"${RAGBASE[@]}" doctor || die "体检发现必须修复的问题（✗），请按上面的提示处理后重新运行"

if [ "$SERVE" -eq 0 ]; then
  step "7/7 已按 --no-serve 跳过启动"
  ok "准备完成。启动命令：uv run ragbase serve"
  exit 0
fi

step "7/7 启动服务（Ctrl+C 停止）"
echo "提示：首次使用请先把文档放进 $RAG_HOME/data/ 并运行 uv run ragbase ingest（另开终端；服务启动前建好索引最方便）"
exec "${RAGBASE[@]}" serve ${SERVE_ARGS[@]+"${SERVE_ARGS[@]}"}
