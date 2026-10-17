import { Button } from '@/components/ui/button'
import { useHealth } from '@/hooks/queries'
import { useAuth } from '@/hooks/useAuth'
import { ApiError, type Health } from '@/lib/api'
import { replaceKey } from '@/lib/auth'
import { relativeTime } from '@/lib/time'
import { cn } from '@/lib/utils'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'

type Tone = 'ok' | 'warn' | 'bad' | 'idle'

const MODE_LABEL: Record<Health['mode'], string> = { vector: '向量', bm25: 'BM25', hybrid: '混合（向量 + BM25）' }
const DOT: Record<Tone, string> = {
  ok: 'bg-success',
  warn: 'bg-warning',
  bad: 'bg-destructive',
  idle: 'bg-muted-foreground/50',
}

function summarize(health: Health | undefined, offline: boolean, locked: boolean): { tone: Tone; label: string } {
  if (locked) return { tone: 'warn', label: '需要密钥' }
  if (offline) return { tone: 'bad', label: '离线' }
  if (!health) return { tone: 'idle', label: '检测中' }
  if (health.status === 'ok' && health.agent_ready) return { tone: 'ok', label: '就绪' }
  if (health.status === 'ok') return { tone: 'warn', label: '仅检索' }
  return { tone: 'bad', label: '未就绪' }
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-6 py-1.5">
      <dt className="shrink-0 text-muted-foreground">{label}</dt>
      <dd className="tabular text-right">{children}</dd>
    </div>
  )
}

export function StatusPill() {
  const { data, isError, error } = useHealth()
  const auth = useAuth()
  const locked = error instanceof ApiError && error.status === 401 // 启用了鉴权但没有有效密钥
  const { tone, label } = summarize(data, isError, locked)
  const health = isError ? undefined : data

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={`服务状态：${label}`}
          className="inline-flex h-8 items-center gap-2 rounded-full border bg-card px-3 text-xs text-foreground transition-colors hover:bg-accent focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
        >
          <span className={cn('size-2 rounded-full', DOT[tone])} aria-hidden />
          {label}
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-72 p-4 text-sm">
        <div className="mb-2 flex items-center gap-2 font-medium">
          <span className={cn('size-2 rounded-full', DOT[tone])} aria-hidden />
          {label}
        </div>
        {health ? (
          <dl className="divide-y divide-border/70">
            <Row label="片段数">{health.chunk_count ?? '—'}</Row>
            <Row label="BM25 构建时间">
              <span title={health.bm25_built_at ?? undefined}>{relativeTime(health.bm25_built_at)}</span>
            </Row>
            <Row label="设备">{health.device}</Row>
            <Row label="召回模式">{MODE_LABEL[health.mode]}</Row>
            <Row label="大模型">
              {health.llm_configured && health.llm_model
                ? `${health.llm_protocol} · ${health.llm_model}`
                : '未配置'}
            </Row>
            <Row label="Agent 就绪">{health.agent_ready ? '是' : '否'}</Row>
          </dl>
        ) : (
          <p className="text-muted-foreground">
            {locked
              ? '服务启用了访问保护，请输入访问密钥后查看状态。'
              : isError
                ? '无法连接后端服务。请确认已启动：uv run ragbase serve（默认端口 8000）。'
                : '正在检测服务状态…'}
          </p>
        )}
        {auth.required && !locked && (
          <div className="mt-3 flex items-center justify-between gap-3 border-t border-border/70 pt-3">
            <span className="text-xs text-muted-foreground">已通过密钥认证</span>
            <Button type="button" variant="outline" size="xs" onClick={replaceKey}>
              更换密钥
            </Button>
          </div>
        )}
        {health && health.status === 'ok' && !health.agent_ready && (
          <p className="mt-2 text-xs text-muted-foreground">大模型未配置，只能检索，暂不能对话。</p>
        )}
        {health && health.status !== 'ok' && (
          <p className="mt-2 text-xs text-muted-foreground">索引不存在或为空：请先建立索引并重启服务。</p>
        )}
      </PopoverContent>
    </Popover>
  )
}
