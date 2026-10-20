import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { ApiError, getDiagnoseJob, startDiagnose, type DiagnoseJob, type DiagnoseRequest } from '@/lib/api'
import { safeStorage } from '@/lib/storage'
import { queryKeys } from './queries'

const ACTIVE_JOB_KEY = 'rag-diagnose-job'
const NOTIFIED_JOB_KEY = 'rag-diagnose-notified-job'

export const isDiagnoseActive = (job: DiagnoseJob | undefined) => job?.status === 'pending' || job?.status === 'running'

function startErrorToast(error: unknown) {
  if (error instanceof ApiError) {
    // 409：入库或体检正在进行；400 / 503：题集为空、模式不可用、流水线未就绪（后端 detail 是中文）
    if (error.status === 409) toast.warning('暂时不能体检', { description: error.message })
    else toast.error('无法开始体检', { description: error.message })
  } else {
    toast.error('无法开始体检')
  }
}

/** 体检任务：记住当前任务 ID（sessionStorage，刷新后恢复进度）、约 1.5 秒轮询到结束；结束时提示、刷新报告列表，成功后回调新报告的 id。 */
export function useDiagnoseJob(onFinished?: (reportId: number) => void) {
  const queryClient = useQueryClient()
  const [jobId, setJobIdState] = useState<string | null>(() => safeStorage.get('session', ACTIVE_JOB_KEY))

  const setJobId = useCallback((id: string | null) => {
    setJobIdState(id)
    safeStorage.set('session', ACTIVE_JOB_KEY, id)
  }, [])

  const job = useQuery({
    queryKey: queryKeys.diagnoseJob(jobId ?? ''),
    queryFn: () =>
      getDiagnoseJob(jobId!).catch((error: unknown) => {
        // 任务不存在（服务重启后任务表清空）：不再追踪
        if (error instanceof ApiError && error.status === 404) setJobId(null)
        throw error
      }),
    enabled: jobId !== null,
    retry: false,
    refetchInterval: (query) => (isDiagnoseActive(query.state.data) ? 1500 : false),
  })

  // 任务结束：每个任务只提示 / 切换一次（刷新页面后不会重复）
  const onFinishedRef = useRef(onFinished)
  useEffect(() => {
    onFinishedRef.current = onFinished
  })
  const lastStatus = useRef<string | null>(null)
  useEffect(() => {
    const data = job.data
    if (!data) return
    const terminal = data.status === 'succeeded' || data.status === 'failed'
    if (!terminal || lastStatus.current === data.status + data.job_id) return
    lastStatus.current = data.status + data.job_id
    if (safeStorage.get('session', NOTIFIED_JOB_KEY) === data.job_id) return
    safeStorage.set('session', NOTIFIED_JOB_KEY, data.job_id)
    if (data.status === 'succeeded') {
      toast.success('体检完成', { description: data.report_id != null ? `报告 #${data.report_id} 已生成` : undefined })
      // 先刷新列表再切换，避免先闪一下旧报告
      void queryClient.invalidateQueries({ queryKey: queryKeys.reports }).then(() => {
        if (data.report_id != null) onFinishedRef.current?.(data.report_id)
      })
    } else {
      toast.error('体检失败', { description: data.error ?? '未知错误' })
    }
  }, [job.data, queryClient])

  const start = useMutation({
    mutationFn: (body: DiagnoseRequest) => startDiagnose(body),
    onSuccess: (started) => {
      queryClient.setQueryData(queryKeys.diagnoseJob(started.job_id), started)
      setJobId(started.job_id)
    },
    onError: startErrorToast,
  })

  return {
    job: job.data,
    dismiss: () => setJobId(null),
    start,
    busy: isDiagnoseActive(job.data) || start.isPending,
  }
}
