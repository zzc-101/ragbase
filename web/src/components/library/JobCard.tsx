import { AlertCircle, CheckCircle2, Loader2, X } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { displayBookName } from '@/lib/books'
import type { Job } from '@/lib/api'
import { relativeTime } from '@/lib/time'

const STATUS = {
  pending: { label: '排队中', variant: 'secondary' as const },
  running: { label: '入库中', variant: 'default' as const },
  succeeded: { label: '已完成', variant: 'secondary' as const },
  failed: { label: '失败', variant: 'destructive' as const },
}

export function JobCard({ job, onDismiss }: { job: Job; onDismiss: () => void }) {
  const active = job.status === 'pending' || job.status === 'running'
  const percent = job.files_total > 0 ? Math.round((job.files_done / job.files_total) * 100) : active ? 0 : 100
  const status = STATUS[job.status]

  return (
    <Card className="gap-0 py-0" aria-live="polite">
      <CardContent className="space-y-3 p-4">
        <div className="flex items-center gap-2.5">
          {active ? (
            <Loader2 className="size-4 animate-spin text-primary" />
          ) : job.status === 'succeeded' ? (
            <CheckCircle2 className="size-4 text-success" />
          ) : (
            <AlertCircle className="size-4 text-destructive" />
          )}
          <span className="text-sm font-medium">入库任务</span>
          <Badge variant={status.variant}>{status.label}</Badge>
          <span className="tabular ml-auto text-xs text-muted-foreground">{relativeTime(job.started_at)}开始</span>
          {!active && (
            <Button variant="ghost" size="icon-xs" aria-label="关闭任务卡片" onClick={onDismiss}>
              <X />
            </Button>
          )}
        </div>

        <div className="space-y-1.5">
          <Progress value={percent} aria-label="入库进度" />
          <div className="flex items-center justify-between gap-4 text-xs text-muted-foreground">
            <span className="min-w-0 truncate" title={job.current_file ?? undefined}>
              {job.current_file ? displayBookName(job.current_file) : active ? '准备中…' : ''}
            </span>
            <span className="tabular shrink-0">
              {job.files_done} / {job.files_total} 个文件
            </span>
          </div>
        </div>

        <p className="tabular flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted-foreground">
          <span>
            新增片段 <b className="font-medium text-foreground">{job.chunks_added}</b>
          </span>
          <span>
            删除片段 <b className="font-medium text-foreground">{job.chunks_deleted}</b>
          </span>
        </p>

        {job.skipped_files.length > 0 && (
          <div className="rounded-lg bg-warning/10 px-3 py-2 text-xs text-warning">
            <p className="font-medium">跳过 {job.skipped_files.length} 个文件</p>
            <ul className="mt-1 space-y-0.5">
              {job.skipped_files.map((file) => (
                <li key={file} className="break-all">
                  {file}
                </li>
              ))}
            </ul>
          </div>
        )}
        {job.error && (
          <div role="alert" className="rounded-lg bg-destructive/10 px-3 py-2 text-xs break-words text-destructive">
            {job.error}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
