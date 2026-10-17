import { HoverCard, HoverCardContent, HoverCardTrigger } from '@/components/ui/hover-card'
import type { Citation } from '@/lib/api'
import { citationLabel } from '@/lib/books'

export function SourceLine({ citation }: { citation: Citation }) {
  const parts = citationLabel(citation.source, citation.heading_path, citation.page)
  return (
    <p className="flex flex-wrap items-center gap-x-1.5 text-xs text-muted-foreground">
      {parts.map((part, index) => (
        <span key={index} className="inline-flex items-center gap-1.5">
          {index > 0 && <span aria-hidden>›</span>}
          <span className={index === 0 ? 'font-medium text-foreground' : undefined}>{part}</span>
        </span>
      ))}
    </p>
  )
}

/** 回答里的引用小标：悬停预览片段，点击在依据面板里定位。找不到对应片段（历史消息没有保存引用）时只显示成普通文字。 */
export function CitationChip({ n, citation, onSelect }: { n: number; citation?: Citation; onSelect?: (n: number) => void }) {
  if (!citation) return <span className="text-muted-foreground">[{n}]</span>
  return (
    <HoverCard openDelay={120} closeDelay={80}>
      <HoverCardTrigger asChild>
        <button
          type="button"
          onClick={() => onSelect?.(n)}
          aria-label={`引用 ${n}：${citation.source}`}
          className="tabular mx-[1px] inline-flex h-[1.15rem] min-w-[1.15rem] -translate-y-px items-center justify-center rounded-md bg-primary/10 px-1 align-baseline text-[11px] leading-none font-medium text-primary transition-colors duration-150 hover:bg-primary/20 focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
        >
          {n}
        </button>
      </HoverCardTrigger>
      <HoverCardContent className="w-80 space-y-2 p-3" align="start">
        <SourceLine citation={citation} />
        <p className="line-clamp-6 text-[13px] leading-relaxed whitespace-pre-line">{citation.text}</p>
      </HoverCardContent>
    </HoverCard>
  )
}
