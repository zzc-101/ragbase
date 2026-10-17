import { AlertCircle, Check, Copy, Library, RotateCcw } from 'lucide-react'
import { lazy, memo, Suspense, useState } from 'react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import type { ChatMessage } from '@/lib/chat-types'
import { copyText } from '@/lib/clipboard'
import { formatDuration } from '@/lib/time'
import { ToolSteps } from './ToolSteps'

// Markdown 渲染（react-markdown + remark）体积较大，按需加载，知识库页不用付这份成本
const AnswerMarkdown = lazy(() => import('./AnswerMarkdown').then((m) => ({ default: m.AnswerMarkdown })))

interface Props {
  message: ChatMessage
  onSelectCitation: (messageId: string, n: number) => void
  onOpenSources: (messageId: string) => void
  onRetry: (messageId: string) => void
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <Button
      variant="ghost"
      size="icon-xs"
      aria-label="复制回答"
      onClick={async () => {
        if (await copyText(text)) {
          setCopied(true)
          setTimeout(() => setCopied(false), 1500)
        } else {
          toast.error('复制失败，请手动选择文字')
        }
      }}
    >
      {copied ? <Check /> : <Copy />}
    </Button>
  )
}

export const MessageView = memo(function MessageView({ message, onSelectCitation, onOpenSources, onRetry }: Props) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-user-bubble px-4 py-2.5 text-[15px] leading-relaxed whitespace-pre-wrap break-words">
          {message.content}
        </div>
      </div>
    )
  }

  const streaming = message.status === 'streaming'
  const hasText = message.content.trim().length > 0
  const hasRunningStep = message.steps.some((step) => step.status === 'running')

  return (
    <article className="min-w-0" aria-busy={streaming}>
      <ToolSteps steps={message.steps} />
      {hasText ? (
        <Suspense fallback={<div className="answer whitespace-pre-wrap">{message.content}</div>}>
          <AnswerMarkdown
            content={message.content}
            citations={message.citations}
            streaming={streaming}
            onSelectCitation={(n) => onSelectCitation(message.id, n)}
          />
        </Suspense>
      ) : (
        streaming &&
        !hasRunningStep && <p className="stream-caret answer text-muted-foreground" aria-label="正在思考" />
      )}

      {message.status === 'stopped' && <p className="mt-2 text-xs text-muted-foreground">已停止生成</p>}

      {message.status === 'error' && (
        <div
          role="alert"
          className="mt-3 flex items-start gap-2.5 rounded-lg border border-destructive/30 bg-destructive/5 px-3.5 py-3 text-sm"
        >
          <AlertCircle className="mt-0.5 size-4 shrink-0 text-destructive" />
          <div className="min-w-0 flex-1">
            <p className="text-destructive">{message.error}</p>
          </div>
          <Button variant="outline" size="xs" onClick={() => onRetry(message.id)}>
            <RotateCcw />
            重试
          </Button>
        </div>
      )}

      {message.status === 'done' && hasText && (
        <footer className="mt-2 flex flex-wrap items-center gap-1 text-xs text-muted-foreground">
          <CopyButton text={message.content} />
          {message.tookMs != null && <span className="tabular px-1">{formatDuration(message.tookMs)}</span>}
          {message.citations.length > 0 && (
            <Button variant="ghost" size="xs" className="text-muted-foreground" onClick={() => onOpenSources(message.id)}>
              <Library />
              依据 <span className="tabular">{message.citations.length}</span>
            </Button>
          )}
        </footer>
      )}
    </article>
  )
})
