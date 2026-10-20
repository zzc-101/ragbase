import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Loader2, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { QuestionTable } from '@/components/diagnose/QuestionTable'
import { ReportCompare } from '@/components/diagnose/ReportCompare'
import { ReportHistory } from '@/components/diagnose/ReportHistory'
import { ReportView } from '@/components/diagnose/ReportView'
import { RunPanel } from '@/components/diagnose/RunPanel'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { queryKeys, useHealth, useQuestions, useReport, useReports } from '@/hooks/queries'
import { useDiagnoseJob } from '@/hooks/useDiagnoseJob'
import { ApiError, deleteReport, type ReportSummary } from '@/lib/api'
import { formatDateTime } from '@/lib/time'

export function DiagnosePage() {
  const queryClient = useQueryClient()
  const health = useHealth()
  const questions = useQuestions()
  const reports = useReports()

  // viewId：用户点选的报告；没点过（或那份已被删除）时显示最新一份
  const [viewId, setViewId] = useState<number | null>(null)
  const [compareIds, setCompareIds] = useState<number[]>([])
  const [pendingDelete, setPendingDelete] = useState<ReportSummary | null>(null)
  const scrollRoot = useRef<HTMLDivElement>(null)
  const compareSection = useRef<HTMLElement>(null)

  const diagnose = useDiagnoseJob(setViewId)

  const list = reports.data
  const effectiveViewId = list ? (list.find((item) => item.id === viewId)?.id ?? list[0]?.id ?? null) : null
  const report = useReport(effectiveViewId)

  // 对比：只保留仍存在的报告，较早的（id 小）放左边
  const comparePair = list ? compareIds.filter((id) => list.some((item) => item.id === id)).sort((a, b) => a - b) : []
  const leftReport = useReport(comparePair.length === 2 ? comparePair[0] : null)
  const rightReport = useReport(comparePair.length === 2 ? comparePair[1] : null)

  const toggleCompare = (id: number) =>
    setCompareIds((current) => (current.includes(id) ? current.filter((x) => x !== id) : [...current, id].slice(-2)))

  // 刚选满两份时，把对比区滚进视野
  const comparing = comparePair.length === 2
  useEffect(() => {
    if (comparing) compareSection.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [comparing])

  const remove = useMutation({
    mutationFn: (id: number) => deleteReport(id),
    onSuccess: (_result, id) => {
      toast.success(`已删除报告 #${id}`)
      setCompareIds((current) => current.filter((x) => x !== id))
      setViewId((current) => (current === id ? null : current))
      queryClient.removeQueries({ queryKey: queryKeys.report(id) })
      void queryClient.invalidateQueries({ queryKey: queryKeys.reports })
    },
    onError: (error, id) => {
      toast.error('删除失败', { description: error instanceof Error ? error.message : undefined })
      if (error instanceof ApiError && error.status === 404) {
        setCompareIds((current) => current.filter((x) => x !== id))
        void queryClient.invalidateQueries({ queryKey: queryKeys.reports })
      }
    },
    onSettled: () => setPendingDelete(null),
  })

  const goQuestions = () => document.getElementById('questions')?.scrollIntoView({ behavior: 'smooth', block: 'start' })

  return (
    <ScrollArea className="h-full" viewportRef={scrollRoot}>
      <div className="mx-auto w-full max-w-5xl space-y-8 px-4 py-8 sm:px-6">
        <header>
          <h1 className="text-xl font-semibold tracking-tight">体检</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            用题集测一遍检索的质量和速度，得到参数现状与调整建议；改过参数后重新体检，再并排对比。
          </p>
        </header>

        <section className="space-y-3">
          <h2 className="text-base font-semibold">运行</h2>
          <RunPanel
            defaultMode={health.data?.mode}
            questionCount={questions.data?.length}
            job={diagnose.job}
            busy={diagnose.busy}
            starting={diagnose.start.isPending}
            onStart={(body) => diagnose.start.mutate(body)}
            onDismiss={diagnose.dismiss}
            onGoQuestions={goQuestions}
          />
        </section>

        <section className="space-y-3">
          <h2 className="text-base font-semibold">报告</h2>
          {reports.isLoading && <div className="h-40 animate-pulse rounded-xl bg-muted" aria-hidden />}
          {reports.isError && (
            <div className="rounded-xl border border-dashed px-6 py-8 text-center text-sm text-muted-foreground">
              {reports.error instanceof Error ? reports.error.message : '加载失败'}
              <div className="mt-3">
                <Button variant="outline" size="sm" onClick={() => void reports.refetch()}>
                  重试
                </Button>
              </div>
            </div>
          )}
          {list && list.length === 0 && (
            <div className="rounded-xl border border-dashed px-6 py-10 text-center text-sm text-muted-foreground">
              还没有体检报告。点上面的“开始体检”生成第一份（命令行 ragbase diagnose 的结果也会出现在这里）。
            </div>
          )}
          {list && list.length > 0 && report.isLoading && <div className="h-40 animate-pulse rounded-xl bg-muted" aria-hidden />}
          {report.isError && (
            <div className="rounded-xl border border-dashed px-6 py-8 text-center text-sm text-muted-foreground">
              无法加载报告：{report.error instanceof Error ? report.error.message : '未知错误'}
              <div className="mt-3">
                <Button variant="outline" size="sm" onClick={() => void report.refetch()}>
                  重试
                </Button>
              </div>
            </div>
          )}
          {report.data && <ReportView report={report.data} />}
        </section>

        {list && list.length > 0 && (
          <section className="space-y-3">
            <h2 className="text-base font-semibold">
              历史 <span className="tabular text-sm font-normal text-muted-foreground">共 {list.length} 份</span>
            </h2>
            <ReportHistory
              reports={list}
              viewingId={effectiveViewId}
              compareIds={comparePair}
              onView={(id) => {
                setViewId(id)
                scrollRoot.current?.scrollTo({ top: 0, behavior: 'smooth' })
              }}
              onToggleCompare={toggleCompare}
              onDelete={setPendingDelete}
            />
          </section>
        )}

        {comparing && (
          <section ref={compareSection} className="scroll-mt-4 space-y-3">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-base font-semibold">对比</h2>
              <Button variant="ghost" size="sm" onClick={() => setCompareIds([])}>
                <X />
                退出对比
              </Button>
            </div>
            {leftReport.data && rightReport.data ? (
              <ReportCompare left={leftReport.data} right={rightReport.data} />
            ) : leftReport.isError || rightReport.isError ? (
              <div className="rounded-xl border border-dashed px-6 py-8 text-center text-sm text-muted-foreground">无法加载报告，请稍后重试。</div>
            ) : (
              <div className="flex h-40 items-center justify-center rounded-xl bg-muted/50 text-sm text-muted-foreground">
                <Loader2 className="mr-2 size-4 animate-spin" />
                加载中…
              </div>
            )}
          </section>
        )}

        <QuestionTable />
      </div>

      <AlertDialog open={pendingDelete !== null} onOpenChange={(open) => !open && !remove.isPending && setPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>删除这份报告？</AlertDialogTitle>
            <AlertDialogDescription>
              将删除报告 #{pendingDelete?.id}（{pendingDelete && formatDateTime(pendingDelete.created_at)}）。此操作无法撤销。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={remove.isPending}>取消</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={remove.isPending}
              onClick={(event) => {
                event.preventDefault()
                if (pendingDelete) remove.mutate(pendingDelete.id)
              }}
            >
              {remove.isPending && <Loader2 className="animate-spin" />}
              删除
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </ScrollArea>
  )
}
