/** 后端 API 的类型化客户端。类型与 api_schemas.py 一一对应；所有路径都在根上（无 /api 前缀）。 */

// ---------------------------------------------------------------- 类型
export interface Scores {
  vec_rank: number | null
  bm25_rank: number | null
  rrf_score: number | null
  rerank_score: number | null
}

export interface Citation {
  n: number
  chunk_id: string
  text: string
  source: string
  page: number | null
  heading_path: string | null
  scores: Scores
}

export interface ToolCallInfo {
  name: string
  args: Record<string, unknown>
}

export interface Health {
  status: 'ok' | 'not_ready'
  chunk_count: number | null
  bm25_ready: boolean
  bm25_built_at: string | null
  bm25_chunk_count: number | null
  device: string
  mode: 'vector' | 'bm25' | 'hybrid'
  llm_configured: boolean
  llm_protocol: string
  llm_model: string | null
  agent_ready: boolean
}

export interface DocumentInfo {
  name: string
  chunk_count: number
  heading_chunks: number
  has_headings: boolean
}

export type JobStatus = 'pending' | 'running' | 'succeeded' | 'failed'

export interface Job {
  job_id: string
  status: JobStatus
  files_total: number
  files_done: number
  current_file: string | null
  chunks_added: number
  chunks_deleted: number
  skipped_files: string[]
  error: string | null
  started_at: string | null
  finished_at: string | null
}

export interface UploadResult extends Job {
  filename: string
}

export interface SessionSummary {
  session_id: string
  title: string
  updated_at: string
  message_count: number
}

export interface SessionMessage {
  role: 'user' | 'assistant' | 'tool'
  content: string
  tool_calls?: ToolCallInfo[] | null
}

export interface ChatRequest {
  session_id?: string
  message: string
  sources?: string[]
}

// ---------------------------------------------------------------- 错误
/** 统一错误：HTTP 错误带后端的 {error, detail}；网络失败 status 为 0、code 为 "network"。 */
export class ApiError extends Error {
  readonly status: number
  readonly code: string

  constructor(status: number, code: string, detail: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }

  get isNetwork() {
    return this.status === 0
  }
}

/** 给聊天区展示的友好文案（错误码来自 HTTP 错误体或 SSE error 事件）；其余情况用后端 detail。 */
export function describeErrorCode(code: string, detail: string, status = 0): string {
  if (code === 'network') return '网络连接失败，请检查服务是否在运行'
  if (status === 503 || code === 'service_unavailable') return '大模型未配置或服务未就绪'
  if (status === 504 || code === 'llm_timeout') return '大模型响应超时，请稍后重试'
  if (status === 502 || code === 'llm_error') return '大模型调用失败，请稍后重试'
  return detail
}

export function describeChatError(error: unknown): string {
  if (error instanceof ApiError) return describeErrorCode(error.code, error.message, error.status)
  return error instanceof Error ? error.message : '未知错误'
}

export async function errorFromResponse(response: Response): Promise<ApiError> {
  let code = 'http_error'
  let detail = response.statusText || `请求失败（${response.status}）`
  try {
    const body = await response.json()
    if (body && typeof body === 'object') {
      if (typeof body.error === 'string') code = body.error
      if (typeof body.detail === 'string') detail = body.detail
    }
  } catch {
    // 非 JSON 响应：沿用状态文本
  }
  return new ApiError(response.status, code, detail)
}

export function networkError(): ApiError {
  return new ApiError(0, 'network', '无法连接服务')
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, init)
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw networkError()
  }
  if (!response.ok) throw await errorFromResponse(response)
  return (await response.json()) as T
}

function jsonInit(method: string, body?: unknown): RequestInit {
  return {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  }
}

// ---------------------------------------------------------------- 接口
export const getHealth = () => request<Health>('/health')

export const getDocuments = () => request<{ documents: DocumentInfo[] }>('/documents').then((r) => r.documents)

export const deleteDocument = (name: string) =>
  request<{ name: string; chunks_deleted: number }>(`/documents/${encodeURIComponent(name)}`, jsonInit('DELETE'))

export const startIngest = () => request<Job>('/ingest', jsonInit('POST'))

export const getJob = (id: string) => request<Job>(`/ingest/jobs/${encodeURIComponent(id)}`)

/** 上传用 XHR 而不是 fetch：fetch 拿不到上传进度，而教材 PDF 动辄上百 MB。 */
export function uploadDocument(file: File, onProgress?: (fraction: number) => void): Promise<UploadResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/documents')
    xhr.responseType = 'json'
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress?.(event.loaded / event.total)
    }
    xhr.onerror = () => reject(networkError())
    xhr.onload = () => {
      const body = xhr.response as Record<string, unknown> | null
      if (xhr.status >= 200 && xhr.status < 300 && body) {
        resolve(body as unknown as UploadResult)
        return
      }
      const code = typeof body?.error === 'string' ? body.error : 'http_error'
      const detail = typeof body?.detail === 'string' ? body.detail : `上传失败（${xhr.status}）`
      reject(new ApiError(xhr.status, code, detail))
    }
    const form = new FormData()
    form.append('file', file)
    xhr.send(form)
  })
}

export const listSessions = () => request<{ sessions: SessionSummary[] }>('/chat/sessions').then((r) => r.sessions)

export const getSession = (id: string) =>
  request<{ session_id: string; messages: SessionMessage[] }>(`/chat/sessions/${encodeURIComponent(id)}`)

export const deleteSession = (id: string) =>
  request<{ session_id: string; deleted: boolean }>(`/chat/sessions/${encodeURIComponent(id)}`, jsonInit('DELETE'))

/** 发起流式对话，返回原始 Response（调用方交给 lib/sse.ts 解析）；HTTP 错误（503 / 400 / 422…）在这里抛出。 */
export async function openChatStream(body: ChatRequest, signal: AbortSignal): Promise<Response> {
  let response: Response
  try {
    response = await fetch('/chat/stream', { ...jsonInit('POST', body), signal })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw networkError()
  }
  if (!response.ok) throw await errorFromResponse(response)
  if (!response.body) throw new ApiError(0, 'network', '浏览器不支持流式响应')
  return response
}
