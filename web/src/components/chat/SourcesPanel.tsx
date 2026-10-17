import { ChevronDown, ChevronUp, PanelRightClose } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { Citation } from '@/lib/api'
import { displayBookName } from '@/lib/books'
import { cn } from '@/lib/utils'

interface Props {
  citations: Citation[]
  selectedN: number | null
  /** 为 true 时说明有可查看的回答但它没有引用（例如拒答），否则是还没选中任何回答 */
  hasAnswer: boolean
  onClose?: () => void
}

function CitationCard({ citation, selected }: { citation: Citation; selected: boolean }) {
  const [expanded, setExpanded] = useState(false)
  const ref = useRef<HTMLLIElement>(null)
  const score = citation.scores.rerank_score

  useEffect(() => {
    if (selected) ref.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [selected])

  const long = citation.text.length > 120
  return (
    <li
      ref={ref}
      className={cn(
        'rounded-xl border bg-card p-3.5 transition-[border-color,box-shadow] duration-200',
        selected && 'border-primary/60 ring-[3px] ring-primary/15',
      )}
    >
      <div className="flex items-start gap-2.5">
        <span className="tabular mt-0.5 grid h-5 min-w-5 shrink-0 place-items-center rounded-md bg-primary/10 px-1 text-[11px] font-medium text-primary">
          {citation.n}
        </span>
        <div className="min-w-0 flex-1 space-y-0.5">
          <Tooltip>
            <TooltipTrigger asChild>
              <p className="truncate text-sm font-medium">{displayBookName(citation.source)}</p>
            </TooltipTrigger>
            <TooltipContent className="max-w-sm break-all">{citation.source}</TooltipContent>
          </Tooltip>
          {citation.heading_path && (
            <p className="text-xs leading-snug text-muted-foreground">{citation.heading_path.replaceAll(' > ', ' › ')}</p>
          )}
          {citation.page != null && <p className="tabular text-xs text-muted-foreground">第 {citation.page} 页</p>}
        </div>
      </div>
      <p className={cn('mt-2.5 text-[13px] leading-relaxed whitespace-pre-line text-foreground/90', !expanded && 'line-clamp-4')}>
        {citation.text}
      </p>
      <div className="mt-2 flex items-center justify-between">
        {long ? (
          <Button variant="ghost" size="xs" className="-ml-2 text-muted-foreground" onClick={() => setExpanded(!expanded)}>
            {expanded ? <ChevronUp /> : <ChevronDown />}
            {expanded ? '收起' : '展开'}
          </Button>
        ) : (
          <span />
        )}
        {score != null && (
          <span className="tabular text-[11px] text-muted-foreground/80" title="精排得分 P(相关)">
            相关度 {score.toFixed(3)}
          </span>
        )}
      </div>
    </li>
  )
}

export function SourcesPanel({ citations, selectedN, hasAnswer, onClose }: Props) {
  return (
    <div className="flex h-full min-h-0 flex-col bg-background">
      <div className="flex h-11 shrink-0 items-center justify-between border-b px-4">
        <h2 className="text-sm font-medium">
          依据 {citations.length > 0 && <span className="tabular text-muted-foreground">· {citations.length}</span>}
        </h2>
        {onClose && (
          <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="收起依据面板">
            <PanelRightClose />
          </Button>
        )}
      </div>
      {citations.length > 0 ? (
        <ScrollArea className="min-h-0 flex-1">
          <ul className="space-y-3 p-4">
            {citations.map((citation) => (
              <CitationCard key={citation.chunk_id + citation.n} citation={citation} selected={citation.n === selectedN} />
            ))}
          </ul>
        </ScrollArea>
      ) : (
        <p className="px-6 py-10 text-center text-sm leading-relaxed text-muted-foreground">
          {hasAnswer ? '这条回答没有引用知识库片段。' : '回答中的引用编号对应的教材原文会显示在这里。点击编号可定位。'}
        </p>
      )}
    </div>
  )
}
