import { useQuery } from '@tanstack/react-query'
import { getDocuments, getHealth, getReport, listQuestions, listReports, listSessions } from '@/lib/api'

export const queryKeys = {
  health: ['health'] as const,
  documents: ['documents'] as const,
  sessions: ['sessions'] as const,
  job: (id: string) => ['job', id] as const,
  questions: ['questions'] as const,
  reports: ['reports'] as const,
  report: (id: number) => ['report', id] as const,
  diagnoseJob: (id: string) => ['diagnose-job', id] as const,
}

/** 服务状态：每 15 秒轮询一次；失败（服务离线）不重试，交给下一次轮询。 */
export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: getHealth,
    refetchInterval: 15_000,
    retry: false,
  })
}

export function useDocuments() {
  return useQuery({ queryKey: queryKeys.documents, queryFn: getDocuments })
}

export function useSessions() {
  return useQuery({ queryKey: queryKeys.sessions, queryFn: listSessions })
}

export function useQuestions() {
  return useQuery({ queryKey: queryKeys.questions, queryFn: listQuestions })
}

export function useReports() {
  return useQuery({ queryKey: queryKeys.reports, queryFn: listReports })
}

/** 单份完整报告：报告生成后不会再变，缓存时间设长；id 为 null 时不请求。 */
export function useReport(id: number | null) {
  return useQuery({
    queryKey: queryKeys.report(id ?? -1),
    queryFn: () => getReport(id!),
    enabled: id !== null,
    staleTime: 5 * 60_000,
  })
}
