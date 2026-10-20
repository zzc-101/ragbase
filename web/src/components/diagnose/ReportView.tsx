import { TriangleAlert } from 'lucide-react'
import type { ReactNode } from 'react'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import type { Report, StageName } from '@/lib/api'
import { fmtMs, fmtPct, fmtRate, fmtScore, GROUP_LABELS, GROUPS, METRICS, MODE_LABELS, ORIGIN_LABELS, paramRows, STAGE_LABELS, STAGES } from '@/lib/report'
import { formatDateTime } from '@/lib/time'

export function ChatWarning({ count, className }: { count: number; className?: string }) {
  return (
    <div className={`flex items-start gap-2 rounded-lg bg-warning/10 px-3 py-2 text-xs text-warning ${className ?? ''}`} role="note">
      <TriangleAlert className="mt-px size-3.5 shrink-0" />
      <span>体检期间有 {count} 次对话请求，耗时可能偏高。</span>
    </div>
  )
}

function Block({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <Card className="gap-0 py-0">
      <CardContent className="space-y-3 p-4">
        <div>
          <h3 className="text-sm font-semibold">{title}</h3>
          {hint && <p className="mt-0.5 text-xs text-muted-foreground">{hint}</p>}
        </div>
        {children}
      </CardContent>
    </Card>
  )
}

const NETWORK_NOTE: Partial<Record<StageName, (report: Report) => string>> = {
  embed: (r) => (r.params.embedding_provider !== 'local' ? '含网络往返' : ''),
  rerank: (r) => (r.params.reranker_provider === 'api' ? '含网络往返' : ''),
}

/** 单份报告：参数 / 质量 / 速度 / 建议四块，内容与终端报告（render_report）一致。 */
export function ReportView({ report }: { report: Report }) {
  const { params, summary, recommendations } = report
  const { quality, speed, questions: counts } = summary
  const top1 = quality.top1_score
  const total = speed.total
  const perCandidate = speed.rerank_ms_per_candidate

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <h3 className="text-base font-semibold">报告 #{report.id}</h3>
        <Badge variant="secondary">{ORIGIN_LABELS[report.origin]}</Badge>
        <Badge variant="outline">{MODE_LABELS[params.mode] ?? params.mode}</Badge>
        <span className="tabular text-xs text-muted-foreground">{formatDateTime(report.generated_at)}</span>
      </div>

      {report.concurrent_chat_requests > 0 && <ChatWarning count={report.concurrent_chat_requests} />}

      <Block title="当前参数">
        <dl className="grid gap-x-8 gap-y-1.5 text-sm sm:grid-cols-2">
          {paramRows(report).map(([label, value]) => (
            <div key={label} className="flex min-w-0 flex-col gap-0.5 border-b border-dashed py-1.5 sm:flex-row sm:items-baseline sm:justify-between sm:gap-3">
              <dt className="text-xs text-muted-foreground sm:shrink-0">{label}</dt>
              <dd className="tabular min-w-0 break-words sm:text-right">{value}</dd>
            </div>
          ))}
        </dl>
      </Block>

      <Block title="质量" hint="词面代理指标，仅在同一题集上做相对比较">
        {quality.candidates === null ? (
          <p className="text-sm text-muted-foreground">没有带参考答案的题，无法计算召回指标。</p>
        ) : (
          <Table className="min-w-[28rem]">
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="pl-0">阶段</TableHead>
                {METRICS.map((metric) => (
                  <TableHead key={metric} className="text-right font-mono">
                    {metric}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {GROUPS.map((group) => {
                const row = quality[group]
                return (
                  <TableRow key={group}>
                    <TableCell className="pl-0 whitespace-nowrap">{GROUP_LABELS[group]}</TableCell>
                    {METRICS.map((metric) => (
                      <TableCell key={metric} className="tabular text-right">
                        {fmtScore(row?.[metric])}
                      </TableCell>
                    ))}
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        )}
        <dl className="grid gap-x-8 gap-y-1.5 text-sm sm:grid-cols-2">
          <KV label="拒答率" value={`${fmtRate(quality.refusal_rate)}（${quality.refused}/${counts.measured}）`} />
          <KV
            label="Top1 精排得分"
            value={
              top1
                ? `最小 ${fmtScore(top1.min)} / 中位数 ${fmtScore(top1.median)} / 最大 ${fmtScore(top1.max)}`
                : '无（未启用精排，或没有召回结果）'
            }
          />
          <KV label="平均交给大模型的片段数" value={quality.avg_sent_docs != null ? quality.avg_sent_docs.toFixed(2) : '—'} />
        </dl>
      </Block>

      <Block title="速度" hint="毫秒；每题一次召回 + 精排">
        <Table className="min-w-[36rem]">
          <TableHeader>
            <TableRow className="hover:bg-transparent">
              <TableHead className="pl-0">阶段</TableHead>
              <TableHead className="text-right">均值</TableHead>
              <TableHead className="text-right">中位数</TableHead>
              <TableHead className="text-right">最大</TableHead>
              <TableHead className="w-44">占比</TableHead>
              <TableHead>备注</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {STAGES.map((name) => {
              const stat = speed.stages[name]
              if (!stat) return null
              return (
                <TableRow key={name}>
                  <TableCell className="pl-0 whitespace-nowrap">{STAGE_LABELS[name]}</TableCell>
                  <TableCell className="tabular text-right">{fmtMs(stat.mean)}</TableCell>
                  <TableCell className="tabular text-right">{fmtMs(stat.median)}</TableCell>
                  <TableCell className="tabular text-right">{fmtMs(stat.max)}</TableCell>
                  <TableCell>
                    <ShareBar pct={stat.share_pct} />
                  </TableCell>
                  <TableCell className="text-xs whitespace-nowrap text-muted-foreground">{NETWORK_NOTE[name]?.(report) ?? ''}</TableCell>
                </TableRow>
              )
            })}
            {total && (
              <TableRow className="font-medium hover:bg-transparent">
                <TableCell className="pl-0 whitespace-nowrap">检索总耗时（召回+精排）</TableCell>
                <TableCell className="tabular text-right">{fmtMs(total.mean)}</TableCell>
                <TableCell className="tabular text-right">{fmtMs(total.median)}</TableCell>
                <TableCell className="tabular text-right">{fmtMs(total.max)}</TableCell>
                <TableCell className="tabular text-xs text-muted-foreground">100.0%</TableCell>
                <TableCell />
              </TableRow>
            )}
          </TableBody>
        </Table>
        {perCandidate != null && (
          <dl className="text-sm">
            <KV
              label="精排每条候选平均"
              value={`${perCandidate.toFixed(2)} ms（平均 ${(speed.avg_candidates ?? 0).toFixed(1)} 条候选）${
                params.reranker_provider === 'api' ? '，含网络往返，云端精排按批计费，仅供参考' : ''
              }`}
            />
          </dl>
        )}
        <p className="text-xs text-muted-foreground">各阶段占比相加可能略低于 100%：差额是阶段之间的少量调度开销。</p>
      </Block>

      <Block title="建议" hint="基于以上数据按固定规则生成，是参考而不是定论">
        {recommendations.length === 0 ? (
          <p className="text-sm text-muted-foreground">未发现明显问题。</p>
        ) : (
          <ol className="grid gap-3 lg:grid-cols-2">
            {recommendations.map((item, index) => (
              <li key={item.id} className="space-y-2 rounded-lg border bg-background/60 p-3.5 text-sm">
                <p className="font-medium">
                  <span className="tabular mr-1.5 text-muted-foreground">{index + 1}.</span>
                  {item.title}
                </p>
                <Row label="现象" text={item.phenomenon} />
                <Row label="可以尝试" text={item.try} />
                <Row label="可能的代价" text={item.cost} />
              </li>
            ))}
          </ol>
        )}
      </Block>

      <p className="text-xs text-muted-foreground">以上都是基于当前题集的相对指标，不是绝对质量；调整参数后重新体检并对比。</p>
    </div>
  )
}

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5 border-b border-dashed py-1.5 sm:flex-row sm:items-baseline sm:justify-between sm:gap-3">
      <dt className="text-xs text-muted-foreground sm:shrink-0">{label}</dt>
      <dd className="tabular min-w-0 break-words sm:text-right">{value}</dd>
    </div>
  )
}

function Row({ label, text }: { label: string; text: string }) {
  return (
    <p className="leading-relaxed">
      <span className="mr-1.5 text-xs font-medium text-primary">{label}</span>
      <span className="text-muted-foreground">{text}</span>
    </p>
  )
}

/** 阶段耗时占比的横向条（纯 Tailwind，不用图表库）。 */
function ShareBar({ pct }: { pct: number }) {
  const width = Math.max(0, Math.min(100, pct))
  return (
    <div className="flex items-center gap-2" role="img" aria-label={`占比 ${fmtPct(pct)}`}>
      <div className="h-2 min-w-20 flex-1 overflow-hidden rounded-full bg-muted">
        <div className="h-full rounded-full bg-primary" style={{ width: `${width}%` }} />
      </div>
      <span className="tabular w-12 shrink-0 text-right text-xs text-muted-foreground">{fmtPct(pct)}</span>
    </div>
  )
}
