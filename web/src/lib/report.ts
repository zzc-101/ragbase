/** 体检报告的展示辅助：标签、格式化、参数行；与 src/ragbase/diagnostics.py 的 render_report 保持一致。 */
import type { DiagnoseMode, MetricGroup, Report, ReportOrigin, StageName } from './api'

export const STAGES: StageName[] = ['embed', 'vector_search', 'bm25', 'fuse', 'rerank']

export const STAGE_LABELS: Record<StageName, string> = {
  embed: '向量化查询（embed）',
  vector_search: '向量库查询',
  bm25: 'BM25 关键词（含分词）',
  fuse: 'RRF 融合 + 组装',
  rerank: '精排（rerank）',
}

export const GROUPS: MetricGroup[] = ['candidates', 'reranked', 'llm_context']

export const GROUP_LABELS: Record<MetricGroup, string> = {
  candidates: '召回 TopK 候选',
  reranked: '精排后',
  llm_context: '交给大模型的片段',
}

export const METRICS = ['point_recall', 'context_precision', 'mrr'] as const

export const MODE_LABELS: Record<DiagnoseMode, string> = {
  hybrid: '混合（hybrid）',
  vector: '向量（vector）',
  bm25: 'BM25 关键词（bm25）',
}

export const ORIGIN_LABELS: Record<ReportOrigin, string> = { web: 'Web', cli: '命令行' }

type Num = number | null | undefined

export const fmtScore = (v: Num, digits = 3) => (v == null ? '—' : v.toFixed(digits))
export const fmtMs = (v: Num) => (v == null ? '—' : v.toFixed(1))
export const fmtPct = (v: Num, digits = 1) => (v == null ? '—' : `${v.toFixed(digits)}%`)
export const fmtRate = (v: Num) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)

const text = (v: string | number | null | undefined, fallback = '—') => (v == null || v === '' ? fallback : String(v))

/** 非 hybrid 模式下 per_route_k 后面的附注；对比时去掉它再比较，避免同一个值因模式不同被误判为不同。 */
export const HYBRID_ONLY_NOTE = '（仅 hybrid 生效）'

/** 报告的“当前参数”行（标签、显示值），对比页用同一份来判断哪些参数不同。 */
export function paramRows(report: Report): [string, string][] {
  const { params, summary } = report
  const counts = summary.questions
  const rows: [string, string][] = [
    ['召回方式', params.mode],
    ['per_route_k（混合每路召回数）', text(params.per_route_k) + (params.mode === 'hybrid' ? '' : HYBRID_ONLY_NOTE)],
    ['retrieve_k（精排候选数）', text(params.retrieve_k)],
    ['rerank_k（精排后保留数）', text(params.rerank_k)],
    ['min_score（低于则拒答）', text(params.min_score, '未设置')],
    ['context_min_score（上下文过滤）', text(params.context_min_score, '未设置')],
    ['embedding', `${params.embedding_provider} / ${params.embedding_model}`],
    [
      'reranker',
      params.reranker_provider === 'none' ? 'none（不精排）' : `${params.reranker_provider} / ${text(params.reranker_model)}`,
    ],
    ['reranker 批大小', params.reranker_batch_size != null ? String(params.reranker_batch_size) : '不适用（非本地精排）'],
    ['计算设备', text(params.device)],
    ['片段数', params.chunks ? String(params.chunks.count) : '未能读取'],
    [
      '片段长度（字符）',
      params.chunks
        ? `均值 ${params.chunks.mean.toFixed(0)} / 中位数 ${params.chunks.median.toFixed(0)} / 最大 ${params.chunks.max}`
        : '—',
    ],
    ['切分默认值', `chunk_size=${params.default_chunk_size}，overlap=${params.default_chunk_overlap}`],
    ['模型加载耗时', `${params.model_load_seconds.toFixed(1)} s`],
    [
      '参与统计的题数',
      `${counts.measured}（有参考答案 ${counts.with_reference}）` +
        (counts.duplicates_skipped ? `，跳过重复 ${counts.duplicates_skipped}` : ''),
    ],
  ]
  return rows
}
