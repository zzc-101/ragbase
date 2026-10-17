import { ArrowUp, Square } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { ScopePopover } from './ScopePopover'

const MAX_LENGTH = 2000 // 与后端 ChatRequest.message 的上限一致

interface Props {
  streaming: boolean
  sources: string[]
  onSourcesChange: (sources: string[]) => void
  onSend: (text: string) => void
  onStop: () => void
}

export function Composer({ streaming, sources, onSourcesChange, onSend, onStop }: Props) {
  const [value, setValue] = useState('')
  const area = useRef<HTMLTextAreaElement>(null)
  const composing = useRef(false)

  // 随内容自动增高（上限由 max-h 控制，超出后内部滚动）
  useEffect(() => {
    const el = area.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [value])

  const canSend = value.trim().length > 0 && !streaming

  const submit = () => {
    if (!canSend) return
    onSend(value)
    setValue('')
  }

  return (
    <div className="shrink-0 bg-background px-4 pt-2 pb-4">
      <div className="mx-auto w-full max-w-[720px]">
        <div className="rounded-2xl border bg-card transition-shadow duration-150 focus-within:border-primary/50 focus-within:ring-[3px] focus-within:ring-ring/20">
          <textarea
            ref={area}
            value={value}
            rows={1}
            maxLength={MAX_LENGTH}
            placeholder="问点什么…"
            aria-label="输入问题"
            onChange={(event) => setValue(event.target.value)}
            onCompositionStart={() => (composing.current = true)}
            onCompositionEnd={() => (composing.current = false)}
            onKeyDown={(event) => {
              // 输入法选词时的回车不发送（Safari 在 compositionend 之后的那次 keydown 的 keyCode 是 229）
              if (event.key !== 'Enter' || event.shiftKey) return
              if (composing.current || event.nativeEvent.isComposing || event.keyCode === 229) return
              event.preventDefault()
              submit()
            }}
            className="block max-h-48 min-h-[2.75rem] w-full resize-none bg-transparent px-4 pt-3 pb-1 text-[15px] leading-relaxed outline-none placeholder:text-muted-foreground"
          />
          <div className="flex items-center justify-between px-2 pb-2">
            <ScopePopover selected={sources} onChange={onSourcesChange} />
            <div className="flex items-center gap-2">
              {value.length > MAX_LENGTH * 0.9 && (
                <span className={cn('tabular text-xs', value.length >= MAX_LENGTH ? 'text-destructive' : 'text-muted-foreground')}>
                  {value.length}/{MAX_LENGTH}
                </span>
              )}
              {streaming ? (
                <Button size="sm" variant="outline" onClick={onStop} aria-label="停止生成">
                  <Square className="fill-current" />
                  停止
                </Button>
              ) : (
                <Button size="icon-sm" onClick={submit} disabled={!canSend} aria-label="发送">
                  <ArrowUp />
                </Button>
              )}
            </div>
          </div>
        </div>
        <p className="mt-2 text-center text-xs text-muted-foreground">Enter 发送，Shift + Enter 换行</p>
      </div>
    </div>
  )
}
