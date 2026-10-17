import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { ApiError, getJob, startIngest, uploadDocument, type Job } from '@/lib/api'
import { safeStorage } from '@/lib/storage'
import { queryKeys } from './queries'

const ACTIVE_JOB_KEY = 'rag-active-job'
const NOTIFIED_JOB_KEY = 'rag-notified-job'

export const isActive = (job: Job | undefined) => job?.status === 'pending' || job?.status === 'running'

function conflictToast(error: unknown) {
  if (error instanceof ApiError && error.status === 409) {
    toast.warning('已有入库任务在运行', { description: error.message })
  } else if (error instanceof ApiError && error.status === 413) {
    toast.error('文件太大', { description: error.message })
  } else {
    toast.error(error instanceof ApiError ? error.message : '操作失败')
  }
}

/** 入库任务：记住当前任务 ID（sessionStorage，刷新后继续显示）、约 1.5 秒轮询到结束、结束时提示并刷新书目与状态。 */
export function useIngestJob() {
  const queryClient = useQueryClient()
  const [jobId, setJobIdState] = useState<string | null>(() => safeStorage.get('session', ACTIVE_JOB_KEY))
  const [uploadProgress, setUploadProgress] = useState<number | null>(null)

  const setJobId = useCallback((id: string | null) => {
    setJobIdState(id)
    safeStorage.set('session', ACTIVE_JOB_KEY, id)
  }, [])

  const job = useQuery({
    queryKey: queryKeys.job(jobId ?? ''),
    queryFn: () =>
      getJob(jobId!).catch((error: unknown) => {
        // 任务不存在（服务重启后任务表清空）：不再追踪
        if (error instanceof ApiError && error.status === 404) setJobId(null)
        throw error
      }),
    enabled: jobId !== null,
    retry: false,
    refetchInterval: (query) => (isActive(query.state.data) ? 1500 : false),
  })

  // 任务结束：每个任务只提示一次（刷新页面后不会重复提示）
  const lastStatus = useRef<string | null>(null)
  useEffect(() => {
    const data = job.data
    if (!data) return
    const terminal = data.status === 'succeeded' || data.status === 'failed'
    if (terminal && lastStatus.current !== data.status + data.job_id) {
      lastStatus.current = data.status + data.job_id
      void queryClient.invalidateQueries({ queryKey: queryKeys.documents })
      void queryClient.invalidateQueries({ queryKey: queryKeys.health })
      if (safeStorage.get('session', NOTIFIED_JOB_KEY) !== data.job_id) {
        safeStorage.set('session', NOTIFIED_JOB_KEY, data.job_id)
        if (data.status === 'succeeded') {
          toast.success('入库完成', { description: `新增 ${data.chunks_added} 个片段，删除 ${data.chunks_deleted} 个` })
        } else {
          toast.error('入库失败', { description: data.error ?? '未知错误' })
        }
      }
    }
  }, [job.data, queryClient])

  const track = useCallback(
    (started: Job) => {
      queryClient.setQueryData(queryKeys.job(started.job_id), started)
      setJobId(started.job_id)
    },
    [queryClient, setJobId],
  )

  const rescan = useMutation({
    mutationFn: startIngest,
    onSuccess: track,
    onError: conflictToast,
  })

  const upload = useMutation({
    mutationFn: (file: File) => {
      setUploadProgress(0)
      return uploadDocument(file, setUploadProgress)
    },
    onSuccess: (result) => {
      toast.success(`已上传 ${result.filename}`, { description: '正在入库…' })
      track(result)
    },
    onError: conflictToast,
    onSettled: () => setUploadProgress(null),
  })

  return {
    job: job.data,
    jobId,
    dismiss: () => setJobId(null),
    rescan,
    upload,
    uploadProgress,
    busy: isActive(job.data) || rescan.isPending || upload.isPending,
  }
}
