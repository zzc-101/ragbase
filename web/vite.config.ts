import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { defineConfig } from 'vite'

// 开发时把 API 请求代理到后端（uv run api.py，默认 8000 端口）。
// 前端路由是 /、/library、/c/:id，与下面的 API 路径前缀互不重叠，所以刷新页面不会被代理走。
const backend = 'http://127.0.0.1:8000'
const apiPaths = ['/health', '/search', '/ask', '/documents', '/ingest', '/chat']

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(import.meta.dirname, 'src') } },
  server: {
    proxy: Object.fromEntries(apiPaths.map((path) => [path, { target: backend }])),
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
