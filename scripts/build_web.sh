#!/usr/bin/env bash
# 构建前端并复制到包目录 src/ragbase/web/dist，使 wheel / Docker 镜像自带 Web 界面。
# 需要 Node 22 与 pnpm。产物同时留在 web/dist（仓库内运行时优先使用它）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGED="$ROOT/src/ragbase/web/dist"

if ! command -v pnpm >/dev/null 2>&1; then
  echo "错误：没有找到 pnpm。安装方式：corepack enable（Node 自带）或 npm i -g pnpm；Node 需要 22 版本。" >&2
  exit 1
fi

cd "$ROOT/web"
pnpm install --frozen-lockfile
pnpm build

if [ ! -f dist/index.html ]; then
  echo "错误：构建后没有找到 web/dist/index.html" >&2
  exit 1
fi

rm -rf "$PACKAGED"
mkdir -p "$(dirname "$PACKAGED")"
cp -R dist "$PACKAGED"
echo "前端已构建：web/dist，并复制到 src/ragbase/web/dist（已被 .gitignore 忽略；构建 wheel 时会被打包）"
