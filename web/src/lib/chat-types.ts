import type { Citation, SessionMessage } from './api'

export interface ToolStep {
  id: string
  name: string
  args: Record<string, unknown>
  status: 'running' | 'done'
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  steps: ToolStep[]
  citations: Citation[]
  status: 'streaming' | 'done' | 'error' | 'stopped'
  error?: string
  tookMs?: number
}

let counter = 0
export function uid(): string {
  counter += 1
  return `${Date.now().toString(36)}-${counter}-${Math.random().toString(36).slice(2, 7)}`
}

/** 工具调用在界面上的一行描述：检索：{query} / 查看书目 */
export function describeStep(step: Pick<ToolStep, 'name' | 'args'>): string {
  if (step.name === 'search_knowledge') {
    const query = typeof step.args.query === 'string' ? step.args.query : ''
    return query ? `检索：${query}` : '检索知识库'
  }
  if (step.name === 'list_documents') return '查看书目'
  return step.name
}

/** 把 GET /chat/sessions/{id} 的扁平消息还原成“用户消息 + 带工具步骤的助手回答”。
 * 历史里没有引用片段（后端只在流式 / 单轮响应里给），所以还原出的回答 citations 为空。 */
export function messagesFromHistory(history: SessionMessage[]): ChatMessage[] {
  const result: ChatMessage[] = []
  let steps: ToolStep[] = []
  for (const message of history) {
    if (message.role === 'user') {
      steps = []
      result.push({ id: uid(), role: 'user', content: message.content, steps: [], citations: [], status: 'done' })
    } else if (message.role === 'assistant') {
      if (message.tool_calls?.length) {
        steps.push(...message.tool_calls.map((call) => ({ id: uid(), name: call.name, args: call.args, status: 'done' as const })))
      } else {
        result.push({ id: uid(), role: 'assistant', content: message.content, steps, citations: [], status: 'done' })
        steps = []
      }
    }
  }
  return result
}
