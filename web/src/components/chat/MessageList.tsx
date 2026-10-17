import { ArrowDown } from 'lucide-react'
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import type { ChatMessage } from '@/lib/chat-types'
import { MessageView } from './MessageView'

interface Props {
  messages: ChatMessage[]
  streaming: boolean
  onSelectCitation: (messageId: string, n: number) => void
  onOpenSources: (messageId: string) => void
  onRetry: (messageId: string) => void
}

const STICK_THRESHOLD = 80 // 距底部多少像素以内算“跟随”

/** 消息列表：流式输出时自动滚到底，用户向上翻看后就不再抢滚动；发新消息时重新跟随。 */
export function MessageList({ messages, streaming, onSelectCitation, onOpenSources, onRetry }: Props) {
  const viewport = useRef<HTMLDivElement>(null)
  const stick = useRef(true)
  const [showJump, setShowJump] = useState(false)

  const scrollToBottom = useCallback((smooth = false) => {
    const el = viewport.current
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: smooth ? 'smooth' : 'auto' })
  }, [])

  const onScroll = useCallback(() => {
    const el = viewport.current
    if (!el) return
    const near = el.scrollHeight - el.scrollTop - el.clientHeight < STICK_THRESHOLD
    stick.current = near
    setShowJump(!near)
  }, [])

  // 用户消息一出现（= 刚发送 / 刚载入会话）就回到底部并重新跟随
  const lastUserId = [...messages].reverse().find((m) => m.role === 'user')?.id
  useEffect(() => {
    stick.current = true
    scrollToBottom()
  }, [lastUserId, scrollToBottom])

  useLayoutEffect(() => {
    if (stick.current) scrollToBottom()
  }, [messages, scrollToBottom])

  return (
    <div className="relative min-h-0 flex-1">
      <ScrollArea className="h-full" viewportRef={viewport} onViewportScroll={onScroll}>
        <div className="mx-auto w-full max-w-[720px] space-y-8 px-3 pt-6 pb-8 sm:px-4" role="log" aria-live="off" aria-label="对话消息">
          {messages.map((message) => (
            <MessageView
              key={message.id}
              message={message}
              onSelectCitation={onSelectCitation}
              onOpenSources={onOpenSources}
              onRetry={onRetry}
            />
          ))}
        </div>
      </ScrollArea>
      {showJump && streaming && (
        <Button
          variant="outline"
          size="sm"
          className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full bg-card shadow-sm"
          onClick={() => {
            stick.current = true
            setShowJump(false)
            scrollToBottom(true)
          }}
        >
          <ArrowDown />
          回到底部
        </Button>
      )}
    </div>
  )
}
