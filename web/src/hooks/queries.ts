import { useQuery } from '@tanstack/react-query'
import { getDocuments, getHealth, listSessions } from '@/lib/api'

export const queryKeys = {
  health: ['health'] as const,
  documents: ['documents'] as const,
  sessions: ['sessions'] as const,
  job: (id: string) => ['job', id] as const,
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
