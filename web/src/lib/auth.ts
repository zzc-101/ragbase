/** 接口鉴权的前端状态：共享密钥（后端 RAG_API_KEY）存在 localStorage，401 时清掉并重新弹出输入框。
 *
 * 这是一个极简的外部 store（不依赖 React），api.ts 在每个请求里读取密钥、遇到 401 / 精简版 /health 时上报，
 * components/AuthDialog.tsx 通过 hooks/useAuth.ts 订阅它。 */
import { safeStorage } from './storage'

const STORAGE_KEY = 'ragbase.apiKey'

export const INVALID_KEY_MESSAGE = '密钥无效或已更换'

export interface AuthState {
  /** 后端启用了鉴权（来自 /health 的 auth_required，或任何一次 401） */
  required: boolean
  /** 密钥输入框是否打开 */
  dialogOpen: boolean
  /** 输入框里的提示（如“密钥无效或已更换”） */
  message: string | null
}

let state: AuthState = { required: false, dialogOpen: false, message: null }
const listeners = new Set<() => void>()

// localStorage 不可用（隐私模式等）时退回内存，本次页面生命周期内仍可用
let memoryKey: string | null | undefined

function setState(patch: Partial<AuthState>) {
  const next = { ...state, ...patch }
  if (next.required === state.required && next.dialogOpen === state.dialogOpen && next.message === state.message) return
  state = next
  listeners.forEach((listener) => listener())
}

export function getAuthState(): AuthState {
  return state
}

export function subscribeAuth(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export function getApiKey(): string | null {
  if (memoryKey === undefined) memoryKey = safeStorage.get('local', STORAGE_KEY) || null
  return memoryKey
}

function storeApiKey(key: string | null) {
  memoryKey = key
  safeStorage.set('local', STORAGE_KEY, key)
}

/** 任何一次请求返回 401：清掉失效的密钥并重新弹出输入框。sentKey 是该请求发出时用的密钥——
 * 如果用户在此期间已经换成了新密钥，迟到的旧请求的 401 不应该再把新密钥清掉。 */
export function reportUnauthorized(sentKey: string | null) {
  if (sentKey !== getApiKey()) return
  if (sentKey) storeApiKey(null)
  setState({ required: true, dialogOpen: true, message: sentKey ? INVALID_KEY_MESSAGE : state.message })
}

/** /health 的结果：full 表示拿到了完整内容（密钥有效，或后端没开鉴权）。 */
export function reportHealth(info: { authRequired: boolean; full: boolean }, sentKey: string | null) {
  if (!info.authRequired) {
    setState({ required: false, dialogOpen: false, message: null })
  } else if (info.full) {
    setState({ required: true })
  } else if (sentKey !== getApiKey()) {
    // 迟到的旧响应，忽略
  } else if (sentKey) {
    reportUnauthorized(sentKey) // 存着的密钥已失效（被更换）
  } else {
    setState({ required: true, dialogOpen: true })
  }
}

/** 状态栏的“更换密钥”：清掉当前密钥并打开输入框。 */
export function replaceKey() {
  storeApiKey(null)
  setState({ required: true, dialogOpen: true, message: null })
}

/** 用 /health 校验密钥：完整内容 = 有效。有效则保存并关闭输入框；返回 null 表示成功，否则是错误文案。 */
export async function connect(key: string): Promise<string | null> {
  const trimmed = key.trim()
  if (!trimmed) return '请输入密钥'
  let body: Record<string, unknown>
  try {
    const response = await fetch('/health', { headers: { Authorization: `Bearer ${trimmed}` } })
    if (!response.ok) return `连接失败（${response.status}）`
    body = (await response.json()) as Record<string, unknown>
  } catch {
    return '无法连接服务'
  }
  if (body.auth_required && !('chunk_count' in body)) return '密钥不正确'
  storeApiKey(trimmed)
  setState({ required: Boolean(body.auth_required), dialogOpen: false, message: null })
  return null
}
