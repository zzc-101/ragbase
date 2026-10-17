import { useQueryClient } from '@tanstack/react-query'
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { useNavigate } from 'react-router'
import { ApiError, describeChatError, describeErrorCode, getSession, openChatStream } from '@/lib/api'
import { messagesFromHistory, uid, type ChatMessage, type ToolStep } from '@/lib/chat-types'
import { chatEvents } from '@/lib/sse'
import { queryKeys, useDocuments } from './queries'

interface ChatContextValue {
  sessionId: string | undefined
  messages: ChatMessage[]
  streaming: boolean
  historyLoading: boolean
  historyError: string | null
  selectedSources: string[]
  setSelectedSources: (sources: string[]) => void
  send: (text: string) => Promise<void>
  stop: () => void
  retry: (messageId: string) => void
  /** 路由里的会话 ID 变化时调用：undefined = 新对话，否则加载该会话。 */
  syncRoute: (routeSessionId: string | undefined) => void
  /** 会话被删除后丢弃本地缓存。 */
  forget: (sessionId: string) => void
}

const ChatContext = createContext<ChatContextValue | null>(null)

const markDone = (steps: ToolStep[]): ToolStep[] =>
  steps.some((step) => step.status === 'running') ? steps.map((step) => ({ ...step, status: 'done' as const })) : steps

/** 对话状态放在路由之上：切到知识库页再回来、/ 与 /c/:id 之间跳转，正在进行的流式输出都不会中断。 */
export function ChatProvider({ children }: { children: ReactNode }) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const documents = useDocuments()

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [sessionId, setSessionIdState] = useState<string | undefined>()
  const [streaming, setStreaming] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [selectedSources, setSelectedSources] = useState<string[]>([])

  const sessionIdRef = useRef<string | undefined>(undefined)
  const controllerRef = useRef<AbortController | null>(null)
  const sourcesRef = useRef<string[]>([])
  const cacheRef = useRef(new Map<string, ChatMessage[]>())
  const loadTokenRef = useRef(0)

  // 已选范围里去掉已被删除的书（否则后端会因未知书名返回 400）
  const validSources = useMemo(() => {
    if (!documents.data) return selectedSources
    const names = new Set(documents.data.map((doc) => doc.name))
    return selectedSources.filter((name) => names.has(name))
  }, [documents.data, selectedSources])
  useEffect(() => {
    sourcesRef.current = validSources
  }, [validSources])

  // 缓存当前会话的消息（含流式得到的引用片段，与 GET 历史里的 citations 同形），切换会话再切回来时不丢
  useEffect(() => {
    if (sessionId && !streaming && !historyLoading && messages.length) cacheRef.current.set(sessionId, messages)
  }, [sessionId, streaming, historyLoading, messages])

  const setSessionId = useCallback((id: string | undefined) => {
    sessionIdRef.current = id
    setSessionIdState(id)
  }, [])

  const send = useCallback(
    async (text: string) => {
      const message = text.trim()
      if (!message || controllerRef.current) return

      const botId = uid()
      setMessages((list) => [
        ...list,
        { id: uid(), role: 'user', content: message, steps: [], citations: [], status: 'done' },
        { id: botId, role: 'assistant', content: '', steps: [], citations: [], status: 'streaming' },
      ])
      const controller = new AbortController()
      controllerRef.current = controller
      setStreaming(true)

      const patch = (update: (m: ChatMessage) => ChatMessage) =>
        setMessages((list) => list.map((m) => (m.id === botId ? update(m) : m)))

      // token 先攒在这里，每帧最多刷新一次界面，避免逐 token 重渲染 Markdown
      let content = ''
      let frame = 0
      const flush = () => {
        frame = 0
        const text = content
        patch((m) => ({ ...m, content: text, steps: markDone(m.steps) }))
      }
      const cancelFrame = () => {
        if (frame) cancelAnimationFrame(frame)
        frame = 0
      }

      let finished = false
      try {
        const sources = sourcesRef.current
        const response = await openChatStream(
          { session_id: sessionIdRef.current, message, sources: sources.length ? sources : undefined },
          controller.signal,
        )
        for await (const event of chatEvents(response.body!)) {
          if (event.type === 'session') {
            setSessionId(event.session_id)
            if (!window.location.pathname.startsWith('/c/')) navigate(`/c/${event.session_id}`, { replace: true })
          } else if (event.type === 'tool_call') {
            cancelFrame()
            content = '' // 调用工具之前模型说的话不是最终回答
            patch((m) => ({
              ...m,
              content: '',
              steps: [...markDone(m.steps), { id: uid(), name: event.name, args: event.args, status: 'running' }],
            }))
          } else if (event.type === 'token') {
            content += event.text
            if (!frame) frame = requestAnimationFrame(flush)
          } else if (event.type === 'citations') {
            patch((m) => ({ ...m, citations: event.citations }))
          } else if (event.type === 'done') {
            cancelFrame()
            finished = true
            patch((m) => ({
              ...m,
              content: event.answer || content, // done.answer 才是权威的最终回答
              steps: markDone(m.steps),
              status: 'done',
              tookMs: event.took_ms,
            }))
          } else if (event.type === 'error') {
            cancelFrame()
            finished = true
            const errorText = describeErrorCode(event.error, event.detail)
            patch((m) => ({ ...m, content, steps: markDone(m.steps), status: 'error', error: errorText }))
          }
        }
        if (!finished) {
          cancelFrame()
          patch((m) => ({ ...m, content, steps: markDone(m.steps), status: 'error', error: '连接中断，请重试' }))
        }
      } catch (error) {
        cancelFrame()
        if (error instanceof DOMException && error.name === 'AbortError') {
          patch((m) => ({ ...m, content, steps: markDone(m.steps), status: 'stopped' }))
        } else {
          const errorText = error instanceof ApiError ? describeChatError(error) : '连接中断，请重试'
          patch((m) => ({ ...m, content, steps: markDone(m.steps), status: 'error', error: errorText }))
        }
      } finally {
        if (controllerRef.current === controller) {
          controllerRef.current = null
          setStreaming(false)
        }
        void queryClient.invalidateQueries({ queryKey: queryKeys.sessions })
      }
    },
    [navigate, queryClient, setSessionId],
  )

  const stop = useCallback(() => controllerRef.current?.abort(), [])

  const retry = useCallback(
    (messageId: string) => {
      if (controllerRef.current) return
      const index = messages.findIndex((m) => m.id === messageId)
      const previous = messages[index - 1]
      if (index < 1 || previous?.role !== 'user') return
      setMessages(messages.slice(0, index - 1))
      void send(previous.content)
    },
    [messages, send],
  )

  const syncRoute = useCallback(
    (routeSessionId: string | undefined) => {
      if (routeSessionId === sessionIdRef.current) return
      controllerRef.current?.abort()
      loadTokenRef.current += 1
      const token = loadTokenRef.current
      setHistoryError(null)
      if (!routeSessionId) {
        setSessionId(undefined)
        setMessages([])
        setHistoryLoading(false)
        return
      }
      setSessionId(routeSessionId)
      const cached = cacheRef.current.get(routeSessionId)
      if (cached) {
        setMessages(cached)
        setHistoryLoading(false)
        return
      }
      setMessages([])
      setHistoryLoading(true)
      getSession(routeSessionId)
        .then((data) => {
          if (loadTokenRef.current !== token) return
          setMessages(messagesFromHistory(data.messages))
        })
        .catch((error: unknown) => {
          if (loadTokenRef.current !== token) return
          const gone = error instanceof ApiError && error.status === 404
          setHistoryError(gone ? '这个会话不存在或已过期（服务重启或闲置过久会清空会话）' : describeChatError(error))
        })
        .finally(() => {
          if (loadTokenRef.current === token) setHistoryLoading(false)
        })
    },
    [setSessionId],
  )

  const forget = useCallback((id: string) => {
    cacheRef.current.delete(id)
  }, [])

  const value = useMemo<ChatContextValue>(
    () => ({
      sessionId,
      messages,
      streaming,
      historyLoading,
      historyError,
      selectedSources: validSources,
      setSelectedSources,
      send,
      stop,
      retry,
      syncRoute,
      forget,
    }),
    [sessionId, messages, streaming, historyLoading, historyError, validSources, send, stop, retry, syncRoute, forget],
  )

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>
}

export function useChat(): ChatContextValue {
  const value = useContext(ChatContext)
  if (!value) throw new Error('useChat 需要在 ChatProvider 内使用')
  return value
}
