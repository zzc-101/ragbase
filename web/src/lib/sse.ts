import type { Citation } from './api'

/** 极简 SSE 解析器：fetch + ReadableStream（EventSource 不能 POST）。
 *
 * 处理：分块边界落在任意位置（含 UTF-8 多字节字符中间、\r\n 被拆开）、多行 data、event 名、注释行、
 * 末尾没有空行结束的最后一个事件。 */

export interface SSEMessage {
  event: string
  data: string
}

/** 逐行喂入的事件拼装器；独立出来便于单测。 */
export class SSEParser {
  private buffer = ''
  private event = ''
  private data: string[] = []

  /** 喂入一段已解码的文本，返回其中已完整的事件。 */
  push(chunk: string): SSEMessage[] {
    this.buffer += chunk
    const out: SSEMessage[] = []
    // 行结束符：\r\n、\n、\r。末尾如果是孤立的 \r，可能是 \r\n 的前半，留到下一块再判断。
    const re = /\r\n|\n|\r/g
    let start = 0
    let match: RegExpExecArray | null
    while ((match = re.exec(this.buffer))) {
      if (match[0] === '\r' && match.index === this.buffer.length - 1) break
      this.line(this.buffer.slice(start, match.index), out)
      start = match.index + match[0].length
    }
    this.buffer = this.buffer.slice(start)
    return out
  }

  /** 流结束：处理残留行，并把未以空行结束的最后一个事件也交出来。 */
  flush(): SSEMessage[] {
    const out: SSEMessage[] = []
    if (this.buffer) this.line(this.buffer.replace(/\r$/, ''), out)
    this.buffer = ''
    this.line('', out)
    return out
  }

  private line(line: string, out: SSEMessage[]) {
    if (line === '') {
      if (this.data.length) out.push({ event: this.event || 'message', data: this.data.join('\n') })
      this.event = ''
      this.data = []
      return
    }
    if (line.startsWith(':')) return // 注释 / 心跳
    const colon = line.indexOf(':')
    const field = colon === -1 ? line : line.slice(0, colon)
    let value = colon === -1 ? '' : line.slice(colon + 1)
    if (value.startsWith(' ')) value = value.slice(1)
    if (field === 'event') this.event = value
    else if (field === 'data') this.data.push(value)
    // id / retry 用不到
  }
}

/** 把响应体解析成事件流；signal 中止时抛 AbortError（由 reader.read() 抛出）。 */
export async function* readSSE(body: ReadableStream<Uint8Array>): AsyncGenerator<SSEMessage> {
  const reader = body.getReader()
  const decoder = new TextDecoder('utf-8')
  const parser = new SSEParser()
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      yield* parser.push(decoder.decode(value, { stream: true }))
    }
    yield* parser.push(decoder.decode())
    yield* parser.flush()
  } finally {
    reader.cancel().catch(() => {})
  }
}

// ---------------------------------------------------------------- 聊天事件

export type ChatEvent =
  | { type: 'session'; session_id: string }
  | { type: 'tool_call'; name: string; args: Record<string, unknown> }
  | { type: 'token'; text: string }
  | { type: 'citations'; citations: Citation[] }
  | { type: 'done'; answer: string; took_ms: number }
  | { type: 'error'; error: string; detail: string }

export async function* chatEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<ChatEvent> {
  for await (const message of readSSE(body)) {
    let data: Record<string, unknown>
    try {
      data = JSON.parse(message.data)
    } catch {
      continue // 忽略无法解析的事件
    }
    switch (message.event) {
      case 'session':
      case 'tool_call':
      case 'token':
      case 'citations':
      case 'done':
      case 'error':
        yield { type: message.event, ...data } as ChatEvent
    }
  }
}
