import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { defineConfig } from 'vite'

// 开发时把 API 请求代理到后端（uv run ragbase serve，默认 8000 端口）。
// 前端路由是 /、/library、/diagnose、/c/:id；/diagnose 同时也是 API 前缀（POST /diagnose、/diagnose/reports…），
// 所以对它的浏览器页面导航（Accept 含 text/html）不走代理，刷新页面才能拿到前端。
const backend = 'http://127.0.0.1:8000'
const apiPaths = ['/health', '/search', '/ask', '/documents', '/ingest', '/chat', '/eval-questions']
const pageBypass = (req: { headers: { accept?: string }; url?: string }) =>
  req.headers.accept?.includes('text/html') ? req.url : undefined

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(import.meta.dirname, 'src') } },
  server: {
    proxy: {
      ...Object.fromEntries(apiPaths.map((path) => [path, { target: backend }])),
      '/diagnose': { target: backend, bypass: pageBypass },
    },
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
