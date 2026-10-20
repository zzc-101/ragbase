import { Trash2 } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import type { ReportSummary } from '@/lib/api'
import { fmtMs, fmtPct, fmtRate, fmtScore, MODE_LABELS, ORIGIN_LABELS } from '@/lib/report'
import { cn } from '@/lib/utils'
import { formatDateTime } from '@/lib/time'

interface ReportHistoryProps {
  reports: ReportSummary[]
  /** 当前查看的报告 id */
  viewingId: number | null
  /** 勾选用于对比的报告 id（最多 2 个） */
  compareIds: number[]
  onView: (id: number) => void
  onToggleCompare: (id: number) => void
  onDelete: (report: ReportSummary) => void
}

export function ReportHistory({ reports, viewingId, compareIds, onView, onToggleCompare, onDelete }: ReportHistoryProps) {
  return (
    <Card className="gap-0 py-0">
      <p className="border-b px-4 py-2.5 text-xs text-muted-foreground">
        点击一行查看报告；勾选两份报告可并排对比（已选 <span className="tabular text-foreground">{compareIds.length}</span> / 2）。
      </p>
      <ul className="max-h-[26rem] divide-y overflow-y-auto">
        {reports.map((report) => {
          const viewing = report.id === viewingId
          const checked = compareIds.includes(report.id)
          return (
            <li key={report.id} className={cn('flex items-center gap-2 px-3 py-2.5 sm:px-4', viewing && 'bg-accent/60')}>
              <Checkbox
                checked={checked}
                onCheckedChange={() => onToggleCompare(report.id)}
                aria-label={`选择报告 #${report.id} 用于对比`}
                className="shrink-0"
              />
              <button
                type="button"
                onClick={() => onView(report.id)}
                aria-current={viewing ? 'true' : undefined}
                className="min-w-0 flex-1 space-y-1 rounded-md px-1 py-0.5 text-left focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
              >
                <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <span className="text-sm font-medium">#{report.id}</span>
                  <span className="tabular text-sm">{formatDateTime(report.created_at)}</span>
                  <Badge variant="secondary">{ORIGIN_LABELS[report.origin]}</Badge>
                  <Badge variant="outline">{MODE_LABELS[report.mode] ?? report.mode}</Badge>
                  <span className="tabular text-xs text-muted-foreground">{report.question_count} 题</span>
                  {report.concurrent_chat_requests > 0 && (
                    <Badge variant="outline" className="border-warning/40 text-warning">
                      期间有对话
                    </Badge>
                  )}
                  {viewing && <Badge>当前查看</Badge>}
                </span>
                <span className="tabular flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-muted-foreground">
                  <span>候选召回 {fmtScore(report.candidates_recall)}</span>
                  <span>精排后召回 {fmtScore(report.reranked_recall)}</span>
                  <span>拒答率 {fmtRate(report.refusal_rate)}</span>
                  <span>检索均值 {fmtMs(report.total_ms_mean)} ms</span>
                  <span>精排占比 {fmtPct(report.rerank_share_pct)}</span>
                </span>
              </button>
              <Button
                variant="ghost"
                size="icon-xs"
                className="shrink-0 text-muted-foreground hover:text-destructive"
                aria-label={`删除报告 #${report.id}`}
                onClick={() => onDelete(report)}
              >
                <Trash2 />
              </Button>
            </li>
          )
        })}
      </ul>
    </Card>
  )
}
