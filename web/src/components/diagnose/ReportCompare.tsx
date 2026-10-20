import { TriangleAlert } from 'lucide-react'
import type { ReactNode } from 'react'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import type { Report } from '@/lib/api'
import { fmtMs, fmtRate, fmtScore, GROUP_LABELS, GROUPS, HYBRID_ONLY_NOTE, METRICS, MODE_LABELS, ORIGIN_LABELS, paramRows, STAGE_LABELS, STAGES } from '@/lib/report'
import { cn } from '@/lib/utils'
import { formatDateTime } from '@/lib/time'

/** 哪个方向算“变好”：up 越大越好（召回），down 越小越好（耗时、拒答率），none 只显示差值不着色。 */
type Better = 'up' | 'down' | 'none'

interface MetricLine {
  label: string
  a: number | null | undefined
  b: number | null | undefined
  format: (value: number | null | undefined) => string
  better: Better
  /** 差值的显示：absolute 为绝对值，percent 额外带相对变化百分比 */
  delta: (diff: number, a: number) => string
}

const signed = (text: string, diff: number) => (diff > 0 ? `+${text}` : diff < 0 ? `−${text}` : text)

const scoreDelta = (diff: number) => signed(Math.abs(diff).toFixed(3), diff)
const rateDelta = (diff: number) => signed(`${Math.abs(diff * 100).toFixed(1)} 个百分点`, diff)
const countDelta = (diff: number) => signed(Math.abs(diff).toFixed(2), diff)
const msDelta = (diff: number, a: number) => {
  const text = `${Math.abs(diff).toFixed(1)} ms`
  const relative = a !== 0 ? `（${signed(`${Math.abs((diff / a) * 100).toFixed(1)}%`, diff)}）` : ''
  return signed(text, diff) + relative
}

function DeltaCell({ line }: { line: MetricLine }) {
  const { a, b } = line
  if (a == null || b == null) return <TableCell className="text-right text-muted-foreground">—</TableCell>
  const diff = b - a
  const same = Math.abs(diff) < 1e-9
  const good = line.better === 'up' ? diff > 0 : line.better === 'down' ? diff < 0 : false
  const tone = same || line.better === 'none' ? 'text-muted-foreground' : good ? 'text-success' : 'text-warning'
  return <TableCell className={cn('tabular text-right whitespace-nowrap', tone)}>{same ? '无变化' : line.delta(diff, a)}</TableCell>
}

function MetricTable({ title, lines, left, right }: { title: string; lines: MetricLine[]; left: Report; right: Report }) {
  return (
    <Table className="min-w-[30rem]">
      <TableHeader>
        <TableRow className="hover:bg-transparent">
          <TableHead className="pl-0">{title}</TableHead>
          <TableHead className="text-right">左 #{left.id}</TableHead>
          <TableHead className="text-right">右 #{right.id}</TableHead>
          <TableHead className="text-right">差值（右 − 左）</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {lines.map((line) => (
          <TableRow key={line.label}>
            <TableCell className="pl-0">{line.label}</TableCell>
            <TableCell className="tabular text-right">{line.format(line.a)}</TableCell>
            <TableCell className="tabular text-right">{line.format(line.b)}</TableCell>
            <DeltaCell line={line} />
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

function qualityLines(a: Report, b: Report): MetricLine[] {
  const lines: MetricLine[] = []
  for (const group of GROUPS) {
    for (const metric of METRICS) {
      lines.push({
        label: `${GROUP_LABELS[group]} · ${metric}`,
        a: a.summary.quality[group]?.[metric],
        b: b.summary.quality[group]?.[metric],
        format: fmtScore,
        better: 'up',
        delta: scoreDelta,
      })
    }
  }
  lines.push(
    { label: '拒答率', a: a.summary.quality.refusal_rate, b: b.summary.quality.refusal_rate, format: fmtRate, better: 'down', delta: rateDelta },
    {
      label: '平均交给大模型的片段数',
      a: a.summary.quality.avg_sent_docs,
      b: b.summary.quality.avg_sent_docs,
      format: (v) => (v == null ? '—' : v.toFixed(2)),
      better: 'none',
      delta: countDelta,
    },
  )
  return lines
}

function speedLines(a: Report, b: Report): MetricLine[] {
  const ms = (label: string, x: number | null | undefined, y: number | null | undefined): MetricLine => ({
    label,
    a: x,
    b: y,
    format: fmtMs,
    better: 'down',
    delta: msDelta,
  })
  const lines = [
    ms('检索总耗时 · 均值', a.summary.speed.total?.mean, b.summary.speed.total?.mean),
    ms('检索总耗时 · 中位数', a.summary.speed.total?.median, b.summary.speed.total?.median),
    ms('检索总耗时 · 最大', a.summary.speed.total?.max, b.summary.speed.total?.max),
  ]
  for (const name of STAGES) {
    const x = a.summary.speed.stages[name]?.mean
    const y = b.summary.speed.stages[name]?.mean
    if (x == null && y == null) continue
    lines.push(ms(`${STAGE_LABELS[name]} · 均值`, x, y))
  }
  lines.push(ms('精排每条候选平均', a.summary.speed.rerank_ms_per_candidate, b.summary.speed.rerank_ms_per_candidate))
  return lines
}

function Notice({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-start gap-2 rounded-lg bg-warning/10 px-3 py-2 text-xs text-warning" role="note">
      <TriangleAlert className="mt-px size-3.5 shrink-0" />
      <span>{children}</span>
    </div>
  )
}

function ReportHead({ side, report }: { side: string; report: Report }) {
  return (
    <div className="min-w-0 space-y-1.5 rounded-lg border bg-background/60 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-muted-foreground">{side}</span>
        <span className="text-sm font-semibold">报告 #{report.id}</span>
        <Badge variant="secondary">{ORIGIN_LABELS[report.origin]}</Badge>
        <Badge variant="outline">{MODE_LABELS[report.params.mode] ?? report.params.mode}</Badge>
      </div>
      <p className="tabular text-xs text-muted-foreground">
        {formatDateTime(report.generated_at)} · {report.summary.questions.measured} 题
      </p>
    </div>
  )
}

/** 两份报告并排对比：左为较早的一份。参数不同的项高亮；质量升高、耗时降低用成功色，反之用警示色。 */
export function ReportCompare({ left, right }: { left: Report; right: Report }) {
  const params = paramRows(left)
  const paramsRight = paramRows(right)
  const digestDiffers = left.question_digest !== right.question_digest
  const originDiffers = left.origin !== right.origin
  const noisy = [left, right].filter((report) => report.concurrent_chat_requests > 0)

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2">
        <ReportHead side="左" report={left} />
        <ReportHead side="右" report={right} />
      </div>

      <div className="space-y-2 empty:hidden">
        {digestDiffers && <Notice>题集不同，质量指标不可直接比较。</Notice>}
        {originDiffers && (
          <Notice>
            两份报告来源不同（{ORIGIN_LABELS[left.origin]} / {ORIGIN_LABELS[right.origin]}），运行环境可能不一样，耗时对比仅供参考。
          </Notice>
        )}
        {noisy.map((report) => (
          <Notice key={report.id}>报告 #{report.id} 体检期间有 {report.concurrent_chat_requests} 次对话请求，耗时可能偏高。</Notice>
        ))}
      </div>

      <Card className="gap-0 py-0">
        <CardContent className="space-y-1 p-4">
          <h3 className="text-sm font-semibold">参数</h3>
          <p className="pb-1 text-xs text-muted-foreground">不同的项已高亮。</p>
          <Table className="min-w-[30rem]">
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="pl-0">参数</TableHead>
                <TableHead>左 #{left.id}</TableHead>
                <TableHead>右 #{right.id}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {params.map(([label, value], index) => {
                const other = paramsRight[index]?.[1] ?? '—'
                const differs = value.replace(HYBRID_ONLY_NOTE, '') !== other.replace(HYBRID_ONLY_NOTE, '')
                return (
                  <TableRow key={label} className={cn(differs && 'bg-warning/10 hover:bg-warning/15')}>
                    <TableCell className="pl-0">{label}</TableCell>
                    <TableCell className={cn('tabular', differs && 'font-medium')}>{value}</TableCell>
                    <TableCell className={cn('tabular', differs && 'font-medium')}>{other}</TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card className="gap-0 py-0">
        <CardContent className="space-y-1 p-4">
          <h3 className="text-sm font-semibold">质量</h3>
          <p className="pb-1 text-xs text-muted-foreground">召回、精确率、MRR 升高为变好（绿），拒答率降低为变好。</p>
          <MetricTable title="指标" lines={qualityLines(left, right)} left={left} right={right} />
        </CardContent>
      </Card>

      <Card className="gap-0 py-0">
        <CardContent className="space-y-1 p-4">
          <h3 className="text-sm font-semibold">速度</h3>
          <p className="pb-1 text-xs text-muted-foreground">毫秒；耗时降低为变好（绿），升高为变差（橙）。</p>
          <MetricTable title="阶段" lines={speedLines(left, right)} left={left} right={right} />
        </CardContent>
      </Card>
    </div>
  )
}
