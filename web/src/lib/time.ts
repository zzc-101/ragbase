const RTF = new Intl.RelativeTimeFormat('zh-CN', { numeric: 'auto' })

/** “3 分钟前”这类相对时间；输入是 ISO 字符串或毫秒时间戳。 */
export function relativeTime(value: string | number | null | undefined, now = Date.now()): string {
  if (value == null) return '—'
  const time = typeof value === 'number' ? value : Date.parse(value)
  if (Number.isNaN(time)) return '—'
  const seconds = Math.round((time - now) / 1000)
  const abs = Math.abs(seconds)
  if (abs < 45) return '刚刚'
  if (abs < 3600) return RTF.format(Math.round(seconds / 60), 'minute')
  if (abs < 86400) return RTF.format(Math.round(seconds / 3600), 'hour')
  if (abs < 86400 * 30) return RTF.format(Math.round(seconds / 86400), 'day')
  return new Date(time).toLocaleDateString('zh-CN')
}

export function formatDuration(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`
}

/** “2026/10/20 14:03”这类本地日期时间；无法解析时原样返回。 */
export function formatDateTime(value: string | null | undefined): string {
  if (!value) return '—'
  const time = Date.parse(value)
  if (Number.isNaN(time)) return value
  return new Date(time).toLocaleString('zh-CN', { hour12: false, year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}
