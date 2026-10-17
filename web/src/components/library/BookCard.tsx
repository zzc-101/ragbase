import { Layers, Trash2 } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { DocumentInfo } from '@/lib/api'
import { displayBookName } from '@/lib/books'

export function BookCard({ doc, onDelete, deleting }: { doc: DocumentInfo; onDelete: () => void; deleting: boolean }) {
  const coverage = doc.chunk_count > 0 ? Math.round((doc.heading_chunks / doc.chunk_count) * 100) : 0
  return (
    <Card className="gap-0 py-0 transition-colors duration-150 hover:border-primary/30">
      <CardContent className="flex h-full flex-col gap-4 p-4">
        <div className="flex items-start justify-between gap-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <h3 className="line-clamp-2 min-h-[2.8em] text-[15px] leading-snug font-medium">{displayBookName(doc.name)}</h3>
            </TooltipTrigger>
            <TooltipContent className="max-w-sm break-all">{doc.name}</TooltipContent>
          </Tooltip>
          <Button
            variant="ghost"
            size="icon-xs"
            className="-mt-0.5 -mr-1 shrink-0 text-muted-foreground hover:text-destructive"
            aria-label={`删除 ${displayBookName(doc.name)}`}
            disabled={deleting}
            onClick={onDelete}
          >
            <Trash2 />
          </Button>
        </div>

        <div className="mt-auto space-y-2.5">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span className="inline-flex items-center gap-1.5">
              <Layers className="size-3.5" />
              <span className="tabular">{doc.chunk_count.toLocaleString()}</span> 个片段
            </span>
            <Badge variant={doc.has_headings ? 'secondary' : 'outline'} className={doc.has_headings ? 'text-primary' : 'text-muted-foreground'}>
              {doc.has_headings ? '有章节' : '无章节'}
            </Badge>
          </div>
          <div className="space-y-1">
            <Progress value={coverage} className="h-1" aria-label="章节覆盖率" />
            <p className="tabular text-[11px] text-muted-foreground">章节覆盖 {coverage}%</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
