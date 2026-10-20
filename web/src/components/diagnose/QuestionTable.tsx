import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Download, Loader2, Pencil, Plus, Trash2, Upload } from 'lucide-react'
import { useRef, useState, type ChangeEvent } from 'react'
import { toast } from 'sonner'
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
import { Card } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { queryKeys, useQuestions } from '@/hooks/queries'
import {
  ApiError,
  createQuestion,
  deleteQuestion,
  exportQuestions,
  importQuestions,
  updateQuestion,
  type EvalQuestion,
  type ImportMode,
  type QuestionErrorItem,
  type QuestionInput,
} from '@/lib/api'
import { downloadBlob } from '@/lib/download'
import { QuestionDialog } from './QuestionDialog'
import { QuestionImportDialog } from './QuestionImportDialog'

const MAX_IMPORT_BYTES = 5 * 1024 * 1024

function validationErrors(error: unknown): QuestionErrorItem[] {
  return error instanceof ApiError && error.code === 'validation_error' ? error.errors : []
}

export function QuestionTable() {
  const queryClient = useQueryClient()
  const questions = useQuestions()
  const items = questions.data

  const fileInput = useRef<HTMLInputElement>(null)
  // 新增 / 编辑：editing 为 undefined 表示对话框关闭，null 表示新增
  const [editing, setEditing] = useState<EvalQuestion | null | undefined>(undefined)
  const [formErrors, setFormErrors] = useState<QuestionErrorItem[]>([])
  const [pendingDelete, setPendingDelete] = useState<EvalQuestion | null>(null)
  const [importFile, setImportFile] = useState<{ name: string; items: unknown; key: number } | null>(null)
  const [importErrors, setImportErrors] = useState<{ detail: string | null; errors: QuestionErrorItem[] }>({ detail: null, errors: [] })

  const refresh = () => queryClient.invalidateQueries({ queryKey: queryKeys.questions })

  const save = useMutation({
    mutationFn: (input: QuestionInput) => (editing ? updateQuestion(editing.id, input) : createQuestion(input)),
    onSuccess: () => {
      toast.success(editing ? '已保存' : '已添加')
      setEditing(undefined)
      void refresh()
    },
    onError: (error) => {
      const errors = validationErrors(error)
      setFormErrors(errors)
      if (errors.length === 0) toast.error('保存失败', { description: error instanceof Error ? error.message : undefined })
      if (error instanceof ApiError && error.status === 404) void refresh()
    },
  })

  const remove = useMutation({
    mutationFn: (id: number) => deleteQuestion(id),
    onSuccess: () => {
      toast.success('已删除')
      void refresh()
    },
    onError: (error) => {
      toast.error('删除失败', { description: error instanceof Error ? error.message : undefined })
      if (error instanceof ApiError && error.status === 404) void refresh()
    },
    onSettled: () => setPendingDelete(null),
  })

  const doImport = useMutation({
    mutationFn: ({ items, mode }: { items: unknown; mode: ImportMode }) => importQuestions(items, mode),
    onSuccess: (result) => {
      const reassigned = result.reassigned.length
      toast.success(`已导入 ${result.imported} 道题`, { description: reassigned ? `其中 ${reassigned} 道因 id 冲突改用了新 id` : undefined })
      setImportFile(null)
      void refresh()
    },
    onError: (error) => {
      const errors = validationErrors(error)
      if (errors.length > 0) {
        setImportErrors({ detail: error instanceof Error ? error.message : null, errors })
      } else {
        // 其余错误（网络、409 等）留在对话框里显示，方便重试
        setImportErrors({ detail: error instanceof Error ? error.message : '导入失败', errors: [] })
      }
    },
  })

  const exporter = useMutation({
    mutationFn: exportQuestions,
    onSuccess: (blob) => downloadBlob(blob, 'eval_questions.json'),
    onError: (error) => toast.error('导出失败', { description: error instanceof Error ? error.message : undefined }),
  })

  async function onPickFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = '' // 允许重复选择同一个文件
    if (!file) return
    if (file.size > MAX_IMPORT_BYTES) {
      toast.error('文件太大', { description: '题集 JSON 不应超过 5 MB' })
      return
    }
    try {
      const parsed: unknown = JSON.parse(await file.text())
      setImportErrors({ detail: null, errors: [] })
      setImportFile({ name: file.name, items: parsed, key: Date.now() })
    } catch {
      toast.error('不是有效的 JSON 文件', { description: file.name })
    }
  }

  const openEditor = (question: EvalQuestion | null) => {
    setFormErrors([])
    setEditing(question)
  }

  return (
    <section id="questions" className="scroll-mt-4 space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold">题集</h2>
          <p className="tabular mt-0.5 text-sm text-muted-foreground">
            {items ? `共 ${items.length} 道题` : questions.isError ? '无法加载题集' : '加载中…'}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <input ref={fileInput} type="file" accept=".json,application/json" className="sr-only" tabIndex={-1} aria-hidden onChange={onPickFile} />
          <Button variant="outline" size="sm" onClick={() => fileInput.current?.click()}>
            <Upload />
            导入
          </Button>
          <Button variant="outline" size="sm" disabled={exporter.isPending || !items?.length} onClick={() => exporter.mutate()}>
            {exporter.isPending ? <Loader2 className="animate-spin" /> : <Download />}
            导出
          </Button>
          <Button size="sm" onClick={() => openEditor(null)}>
            <Plus />
            新增题目
          </Button>
        </div>
      </div>

      {questions.isError && (
        <div className="rounded-xl border border-dashed px-6 py-8 text-center text-sm text-muted-foreground">
          {questions.error instanceof Error ? questions.error.message : '加载失败'}
          <div className="mt-3">
            <Button variant="outline" size="sm" onClick={() => void questions.refetch()}>
              重试
            </Button>
          </div>
        </div>
      )}
      {questions.isLoading && <div className="h-32 animate-pulse rounded-xl bg-muted" aria-hidden />}
      {items && items.length === 0 && (
        <div className="rounded-xl border border-dashed px-6 py-10 text-center text-sm text-muted-foreground">
          题集还是空的。点“新增题目”逐条添加，或“导入”一个 JSON 文件（格式同导出文件）。
        </div>
      )}
      {items && items.length > 0 && (
        <Card className="gap-0 py-0">
          {/* 几十条题：表格自身限高并可滚动（纵向 + 横向），表头吸顶 */}
          <Table containerClassName="max-h-[26rem] overflow-y-auto" className="min-w-[44rem]">
            <TableHeader className="sticky top-0 z-10 bg-card">
              <TableRow className="hover:bg-transparent">
                <TableHead className="w-12 pl-4">id</TableHead>
                <TableHead className="min-w-56">问题</TableHead>
                <TableHead className="min-w-44">参考答案</TableHead>
                <TableHead>难度</TableHead>
                <TableHead>题型</TableHead>
                <TableHead>出处</TableHead>
                <TableHead className="w-20 pr-4" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {items.map((item) => (
                <TableRow key={item.id}>
                  <TableCell className="tabular pl-4 text-muted-foreground">{item.id}</TableCell>
                  <TableCell className="align-top">
                    <p className="line-clamp-2 break-words" title={item.question}>
                      {item.question}
                    </p>
                  </TableCell>
                  <TableCell className="align-top">
                    {item.reference ? (
                      <p className="line-clamp-2 break-words text-muted-foreground" title={item.reference}>
                        {item.reference}
                      </p>
                    ) : (
                      <span className="text-muted-foreground/60">—</span>
                    )}
                  </TableCell>
                  <TableCell className="align-top whitespace-nowrap text-muted-foreground">{item.difficulty ?? '—'}</TableCell>
                  <TableCell className="align-top whitespace-nowrap text-muted-foreground">{item.type ?? '—'}</TableCell>
                  <TableCell className="max-w-40 align-top text-muted-foreground">
                    <p className="truncate" title={item.source_topic ?? undefined}>
                      {item.source_topic ?? '—'}
                    </p>
                  </TableCell>
                  <TableCell className="pr-4 align-top">
                    <div className="flex justify-end gap-0.5">
                      <Button variant="ghost" size="icon-xs" aria-label={`编辑题目 #${item.id}`} onClick={() => openEditor(item)}>
                        <Pencil />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon-xs"
                        className="text-muted-foreground hover:text-destructive"
                        aria-label={`删除题目 #${item.id}`}
                        onClick={() => setPendingDelete(item)}
                      >
                        <Trash2 />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>
      )}

      <QuestionDialog
        open={editing !== undefined}
        onOpenChange={(open) => !open && setEditing(undefined)}
        question={editing ?? null}
        pending={save.isPending}
        errors={formErrors}
        onSubmit={(input) => save.mutate(input)}
      />

      <QuestionImportDialog
        key={importFile?.key ?? 'closed'}
        file={importFile}
        existingCount={items?.length ?? 0}
        pending={doImport.isPending}
        errors={importErrors.errors}
        errorDetail={importErrors.detail}
        onImport={(mode) => {
          if (importFile) doImport.mutate({ items: importFile.items, mode })
        }}
        onClose={() => setImportFile(null)}
      />

      <AlertDialog open={pendingDelete !== null} onOpenChange={(open) => !open && !remove.isPending && setPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>删除这道题？</AlertDialogTitle>
            <AlertDialogDescription className="break-words">
              将删除题目 #{pendingDelete?.id}：{pendingDelete?.question.slice(0, 80)}
              {pendingDelete && pendingDelete.question.length > 80 ? '…' : ''}。已有的体检报告不受影响。
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
    </section>
  )
}
