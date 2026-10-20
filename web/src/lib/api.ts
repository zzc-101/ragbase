/** 后端 API 的类型化客户端。类型与 src/ragbase/api/schemas.py 一一对应；所有路径都在根上（无 /api 前缀）。 */

import { getApiKey, reportHealth, reportUnauthorized } from './auth'

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

/** embedding / reranker 的提供方与模型：embedding 为 local | openai；reranker 为 local | api | none（不精排时 model 为 null）。 */
export interface ModelInfo {
  provider: string
  model: string | null
}

export interface Health {
  status: 'ok' | 'not_ready'
  chunk_count: number | null
  bm25_ready: boolean
  bm25_built_at: string | null
  bm25_chunk_count: number | null
  /** 本地模型的计算设备；embedding 与 reranker 都用云端时为 "cloud" */
  device: string
  embedding: ModelInfo
  reranker: ModelInfo
  mode: 'vector' | 'bm25' | 'hybrid'
  llm_configured: boolean
  llm_protocol: string
  llm_model: string | null
  agent_ready: boolean
  /** 后端是否启用了接口鉴权（RAG_API_KEY） */
  auth_required: boolean
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
  /** 助手消息正文里标注了 [n] 的片段（会话级编号）；其他角色为空 */
  citations?: Citation[]
}

export interface ChatRequest {
  session_id?: string
  message: string
  sources?: string[]
}

// ---- 评测题集
export interface EvalQuestion {
  id: number
  question: string
  reference: string | null
  difficulty: string | null
  type: string | null
  source_topic: string | null
  created_at: string
  updated_at: string
}

/** 新增 / 修改的请求体；可选字段传 null 表示清空。 */
export interface QuestionInput {
  question: string
  reference?: string | null
  difficulty?: string | null
  type?: string | null
  source_topic?: string | null
}

/** 题集校验错误（422）。index 从 0 开始，整体错误以及新增 / 修改时为 null。 */
export interface QuestionErrorItem {
  index: number | null
  field: string | null
  message: string
}

export type ImportMode = 'append' | 'replace'

export interface ImportResult {
  imported: number
  /** [旧 id, 新 id]：追加时 id 与已有题冲突而改用了新 id */
  reassigned: [number, number][]
}

// ---- 体检
export type DiagnoseMode = 'vector' | 'bm25' | 'hybrid'

export interface DiagnoseRequest {
  mode?: DiagnoseMode
  limit?: number
}

export interface DiagnoseJob {
  job_id: string
  status: JobStatus
  total: number
  done: number
  report_id: number | null
  error: string | null
  started_at: string | null
  finished_at: string | null
}

export type ReportOrigin = 'cli' | 'web'

export interface ReportSummary {
  id: number
  created_at: string
  origin: ReportOrigin
  mode: DiagnoseMode
  question_count: number
  candidates_recall: number | null
  reranked_recall: number | null
  refusal_rate: number | null
  total_ms_mean: number | null
  rerank_share_pct: number | null
  concurrent_chat_requests: number
}

export interface ChunkStats {
  count: number
  mean: number
  median: number
  max: number
}

export interface ReportParams {
  mode: DiagnoseMode
  per_route_k: number
  retrieve_k: number
  rerank_k: number
  min_score: number | null
  context_min_score: number | null
  embedding_provider: string
  embedding_model: string
  reranker_provider: string
  reranker_model: string | null
  reranker_batch_size: number | null
  device: string
  chunks: ChunkStats | null
  default_chunk_size: number
  default_chunk_overlap: number
  model_load_seconds: number
}

export interface MetricRow {
  point_recall: number
  context_precision: number
  mrr: number
}

export interface ValueStat {
  mean: number
  median: number
  max: number
  min: number
  count: number
}

export interface StageStat extends ValueStat {
  share_pct: number
}

export type StageName = 'embed' | 'vector_search' | 'bm25' | 'fuse' | 'rerank'
export type MetricGroup = 'candidates' | 'reranked' | 'llm_context'

export interface ReportData {
  questions: { measured: number; with_reference: number; duplicates_skipped: number }
  quality: Record<MetricGroup, MetricRow | null> & {
    context_filter: {
      questions: number
      reranked_point_recall: number | null
      llm_point_recall: number | null
      gap: number | null
    } | null
    refused: number
    refusal_rate: number
    top1_score: ValueStat | null
    avg_sent_docs: number | null
  }
  speed: {
    total: ValueStat | null
    stages: Partial<Record<StageName, StageStat>>
    avg_candidates: number | null
    rerank_ms_per_candidate: number | null
  }
}

export interface Recommendation {
  id: string
  title: string
  phenomenon: string
  try: string
  cost: string
}

export interface Report {
  id: number
  origin: ReportOrigin
  generated_at: string
  params: ReportParams
  warmup_question: string | null
  /** 逐题记录（页面不展示） */
  questions: Record<string, unknown>[]
  summary: ReportData
  recommendations: Recommendation[]
  question_ids: unknown[]
  question_digest: string | null
  concurrent_chat_requests: number
}

// ---------------------------------------------------------------- 错误
/** 统一错误：HTTP 错误带后端的 {error, detail}；网络失败 status 为 0、code 为 "network"。 */
export class ApiError extends Error {
  readonly status: number
  readonly code: string
  /** 题集校验失败（422 validation_error）时的逐条错误；其他错误为空数组 */
  readonly errors: QuestionErrorItem[]

  constructor(status: number, code: string, detail: string, errors: QuestionErrorItem[] = []) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.errors = errors
  }

  get isNetwork() {
    return this.status === 0
  }
}

/** 给聊天区展示的友好文案（错误码来自 HTTP 错误体或 SSE error 事件）；其余情况用后端 detail。 */
export function describeErrorCode(code: string, detail: string, status = 0): string {
  if (code === 'network') return '网络连接失败，请检查服务是否在运行'
  if (status === 401 || code === 'unauthorized') return '访问密钥无效或已更换，请在弹出的窗口里重新输入密钥后重试'
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
  let errors: QuestionErrorItem[] = []
  try {
    const body = await response.json()
    if (body && typeof body === 'object') {
      if (typeof body.error === 'string') code = body.error
      if (typeof body.detail === 'string') detail = body.detail
      if (code === 'validation_error' && Array.isArray(body.errors)) errors = body.errors as QuestionErrorItem[]
    }
  } catch {
    // 非 JSON 响应：沿用状态文本
  }
  return new ApiError(response.status, code, detail, errors)
}

export function networkError(): ApiError {
  return new ApiError(0, 'network', '无法连接服务')
}

/** 所有 API 请求（含 SSE）的唯一出口：自动带上 Authorization: Bearer <密钥>，401 时清掉失效密钥并重新弹出输入框。
 * 新增接口请走 request() / apiFetch()，不要直接用 fetch。 */
export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const key = getApiKey()
  const headers = new Headers(init.headers)
  if (key) headers.set('Authorization', `Bearer ${key}`)
  let response: Response
  try {
    response = await fetch(path, { ...init, headers })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw networkError()
  }
  if (response.status === 401) reportUnauthorized(key)
  return response
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await apiFetch(path, init)
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
/** 启用鉴权后，没有有效密钥时 /health 只返回 {status, auth_required}：此时上报给鉴权 store（弹出输入框）并抛 401。 */
export async function getHealth(): Promise<Health> {
  const sentKey = getApiKey()
  const response = await apiFetch('/health')
  if (!response.ok) throw await errorFromResponse(response)
  const body = (await response.json()) as Partial<Health> & { auth_required?: boolean }
  const authRequired = Boolean(body.auth_required)
  const full = 'chunk_count' in body
  reportHealth({ authRequired, full }, sentKey)
  if (authRequired && !full) throw new ApiError(401, 'unauthorized', '需要访问密钥')
  return body as Health
}

export const getDocuments = () => request<{ documents: DocumentInfo[] }>('/documents').then((r) => r.documents)

export const deleteDocument = (name: string) =>
  request<{ name: string; chunks_deleted: number }>(`/documents/${encodeURIComponent(name)}`, jsonInit('DELETE'))

export const startIngest = () => request<Job>('/ingest', jsonInit('POST'))

export const getJob = (id: string) => request<Job>(`/ingest/jobs/${encodeURIComponent(id)}`)

/** 上传用 XHR 而不是 fetch：fetch 拿不到上传进度，而教材 PDF 动辄上百 MB。 */
export function uploadDocument(file: File, onProgress?: (fraction: number) => void): Promise<UploadResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    const key = getApiKey()
    xhr.open('POST', '/documents')
    if (key) xhr.setRequestHeader('Authorization', `Bearer ${key}`)
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
      if (xhr.status === 401) reportUnauthorized(key)
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
  const response = await apiFetch('/chat/stream', { ...jsonInit('POST', body), signal })
  if (!response.ok) throw await errorFromResponse(response)
  if (!response.body) throw new ApiError(0, 'network', '浏览器不支持流式响应')
  return response
}

// ---- 评测题集
export const listQuestions = () => request<{ items: EvalQuestion[] }>('/eval-questions').then((r) => r.items)

export const createQuestion = (body: QuestionInput) => request<EvalQuestion>('/eval-questions', jsonInit('POST', body))

export const updateQuestion = (id: number, body: QuestionInput) =>
  request<EvalQuestion>(`/eval-questions/${id}`, jsonInit('PUT', body))

export const deleteQuestion = (id: number) => request<{ deleted: boolean }>(`/eval-questions/${id}`, jsonInit('DELETE'))

export const importQuestions = (items: unknown, mode: ImportMode) =>
  request<ImportResult>('/eval-questions/import', jsonInit('POST', { items, mode }))

/** 导出要带鉴权头，所以先用 apiFetch 拿到 blob，再由调用方触发下载。 */
export async function exportQuestions(): Promise<Blob> {
  const response = await apiFetch('/eval-questions/export')
  if (!response.ok) throw await errorFromResponse(response)
  return response.blob()
}

// ---- 体检
export const startDiagnose = (body: DiagnoseRequest) => request<DiagnoseJob>('/diagnose', jsonInit('POST', body))

export const getDiagnoseJob = (id: string) => request<DiagnoseJob>(`/diagnose/jobs/${encodeURIComponent(id)}`)

export const listReports = () => request<{ items: ReportSummary[] }>('/diagnose/reports').then((r) => r.items)

export const getReport = (id: number) => request<Report>(`/diagnose/reports/${id}`)

export const deleteReport = (id: number) => request<{ deleted: boolean }>(`/diagnose/reports/${id}`, jsonInit('DELETE'))
