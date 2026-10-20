import { AlertCircle, CheckCircle2, Loader2, Play, X } from 'lucide-react'
import { useState } from 'react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { NativeSelect } from '@/components/ui/native-select'
import { Progress } from '@/components/ui/progress'
import type { DiagnoseJob, DiagnoseMode, DiagnoseRequest } from '@/lib/api'
import { MODE_LABELS } from '@/lib/report'
import { relativeTime } from '@/lib/time'

const STATUS = {
  pending: { label: '排队中', variant: 'secondary' as const },
  running: { label: '体检中', variant: 'default' as const },
  succeeded: { label: '已完成', variant: 'secondary' as const },
  failed: { label: '失败', variant: 'destructive' as const },
}

interface RunPanelProps {
  /** /health 的默认召回方式；未加载时为 undefined */
  defaultMode: DiagnoseMode | undefined
  /** 题集题数；加载中为 undefined */
  questionCount: number | undefined
  job: DiagnoseJob | undefined
  busy: boolean
  starting: boolean
  onStart: (body: DiagnoseRequest) => void
  onDismiss: () => void
  onGoQuestions: () => void
}

export function RunPanel({ defaultMode, questionCount, job, busy, starting, onStart, onDismiss, onGoQuestions }: RunPanelProps) {
  const [mode, setMode] = useState<DiagnoseMode | null>(null)
  const [limitText, setLimitText] = useState('')
  const effectiveMode = mode ?? defaultMode ?? 'hybrid'

  const trimmed = limitText.trim()
  const limit = trimmed === '' ? undefined : Number(trimmed)
  const limitInvalid = limit !== undefined && (!Number.isInteger(limit) || limit < 1)
  const emptyQuestions = questionCount === 0
  const disabled = busy || emptyQuestions || questionCount === undefined || limitInvalid

  const active = job?.status === 'pending' || job?.status === 'running'
  const percent = job && job.total > 0 ? Math.round((job.done / job.total) * 100) : job?.status === 'succeeded' ? 100 : 0

  return (
    <Card className="gap-0 py-0">
      <CardContent className="space-y-4 p-4">
        <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] sm:items-end">
          <label className="space-y-1.5 text-sm">
            <span className="text-muted-foreground">召回方式</span>
            <NativeSelect value={effectiveMode} disabled={busy} onChange={(event) => setMode(event.target.value as DiagnoseMode)}>
              {(Object.keys(MODE_LABELS) as DiagnoseMode[]).map((key) => (
                <option key={key} value={key}>
                  {MODE_LABELS[key]}
                </option>
              ))}
            </NativeSelect>
          </label>
          <label className="space-y-1.5 text-sm">
            <span className="text-muted-foreground">题数上限（留空用全部）</span>
            <Input
              type="number"
              inputMode="numeric"
              min={1}
              step={1}
              placeholder={questionCount ? `全部 ${questionCount} 道` : '全部'}
              value={limitText}
              disabled={busy}
              aria-invalid={limitInvalid || undefined}
              onChange={(event) => setLimitText(event.target.value)}
            />
          </label>
          <Button
            disabled={disabled}
            onClick={() => onStart({ mode: effectiveMode, ...(limit !== undefined ? { limit } : {}) })}
            className="sm:min-w-28"
          >
            {starting || active ? <Loader2 className="animate-spin" /> : <Play />}
            {active ? '体检中…' : '开始体检'}
          </Button>
        </div>

        {limitInvalid && <p className="text-xs text-destructive">题数上限需要是不小于 1 的整数。</p>}

        {emptyQuestions ? (
          <p className="rounded-lg bg-warning/10 px-3 py-2 text-xs text-warning">
            题集是空的，无法体检。请先在下方
            <button type="button" className="mx-0.5 underline underline-offset-2" onClick={onGoQuestions}>
              题集
            </button>
            添加或导入题目。
          </p>
        ) : (
          <p className="text-xs text-muted-foreground">
            体检只测检索（召回 + 精排），不调用大模型。<b className="font-medium text-foreground">体检期间对话会变慢</b>，且不能同时入库。
          </p>
        )}

        {job && (
          <div className="space-y-2.5 border-t pt-4" aria-live="polite">
            <div className="flex items-center gap-2.5">
              {active ? (
                <Loader2 className="size-4 animate-spin text-primary" />
              ) : job.status === 'succeeded' ? (
                <CheckCircle2 className="size-4 text-success" />
              ) : (
                <AlertCircle className="size-4 text-destructive" />
              )}
              <span className="text-sm font-medium">体检任务</span>
              <Badge variant={STATUS[job.status].variant}>{STATUS[job.status].label}</Badge>
              <span className="tabular ml-auto text-xs text-muted-foreground">{relativeTime(job.started_at)}开始</span>
              {!active && (
                <Button variant="ghost" size="icon-xs" aria-label="关闭任务提示" onClick={onDismiss}>
                  <X />
                </Button>
              )}
            </div>
            <Progress value={percent} aria-label="体检进度" />
            <div className="flex items-center justify-between gap-4 text-xs text-muted-foreground">
              <span>{active ? '体检期间对话会变慢，请耐心等待' : job.status === 'succeeded' ? `报告 #${job.report_id ?? '—'} 已生成` : ''}</span>
              <span className="tabular shrink-0">
                {job.done} / {job.total} 道题
              </span>
            </div>
            {job.error && (
              <div role="alert" className="rounded-lg bg-destructive/10 px-3 py-2 text-xs break-words text-destructive">
                {job.error}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
