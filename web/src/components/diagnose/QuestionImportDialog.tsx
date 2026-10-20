import { Loader2 } from 'lucide-react'
import { useState } from 'react'
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
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import type { ImportMode, QuestionErrorItem } from '@/lib/api'
import { cn } from '@/lib/utils'

interface QuestionImportDialogProps {
  /** 已读入的文件；null 表示对话框关闭 */
  file: { name: string; items: unknown } | null
  /** 现有题数（替换时提示将被清空多少道） */
  existingCount: number
  pending: boolean
  /** 服务端校验（422）返回的逐题错误 */
  errors: QuestionErrorItem[]
  errorDetail: string | null
  onImport: (mode: ImportMode) => void
  onClose: () => void
}

const MODES: { value: ImportMode; title: string; hint: string }[] = [
  { value: 'append', title: '追加', hint: '保留现有题目；导入的题 id 与现有题冲突时自动改用新 id' },
  { value: 'replace', title: '替换全部', hint: '先清空现有题集，再导入文件里的题目' },
]

function describeItems(items: unknown): string {
  return Array.isArray(items) ? `${items.length} 道题` : '内容不是题目数组'
}

export function QuestionImportDialog({ file, existingCount, pending, errors, errorDetail, onImport, onClose }: QuestionImportDialogProps) {
  const [mode, setMode] = useState<ImportMode>('append')
  const [confirmReplace, setConfirmReplace] = useState(false)

  return (
    <>
      <Dialog open={file !== null} onOpenChange={(open) => !open && !pending && onClose()}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>导入题集</DialogTitle>
            <DialogDescription className="break-all">
              {file?.name}：{file ? describeItems(file.items) : ''}。有任何一题不合法时整体不写入。
            </DialogDescription>
          </DialogHeader>

          <fieldset className="grid gap-2" disabled={pending}>
            <legend className="sr-only">导入方式</legend>
            {MODES.map((item) => (
              <label
                key={item.value}
                className={cn(
                  'flex cursor-pointer items-start gap-2.5 rounded-lg border p-3 text-sm transition-colors',
                  mode === item.value ? 'border-primary bg-primary/5' : 'hover:bg-muted/50',
                )}
              >
                <input
                  type="radio"
                  name="import-mode"
                  value={item.value}
                  checked={mode === item.value}
                  onChange={() => setMode(item.value)}
                  className="mt-1 accent-primary"
                />
                <span>
                  <span className="font-medium">{item.title}</span>
                  <span className="mt-0.5 block text-xs text-muted-foreground">{item.hint}</span>
                </span>
              </label>
            ))}
          </fieldset>

          {(errors.length > 0 || errorDetail) && (
            <div role="alert" className="space-y-1.5 rounded-lg bg-destructive/10 px-3 py-2 text-xs text-destructive">
              <p className="font-medium">{errorDetail ?? '导入失败'}</p>
              {errors.length > 0 && (
                <ul className="max-h-48 space-y-0.5 overflow-y-auto">
                  {errors.map((item, index) => (
                    <li key={index} className="break-words">
                      {item.index !== null ? `第 ${item.index + 1} 题` : '整体'}
                      {item.field ? `（${item.field}）` : ''}：{item.message}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          <DialogFooter>
            <Button variant="outline" disabled={pending} onClick={onClose}>
              取消
            </Button>
            <Button
              disabled={pending || !Array.isArray(file?.items)}
              variant={mode === 'replace' ? 'destructive' : 'default'}
              onClick={() => (mode === 'replace' ? setConfirmReplace(true) : onImport(mode))}
            >
              {pending && <Loader2 className="animate-spin" />}
              {mode === 'replace' ? '替换并导入' : '导入'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={confirmReplace} onOpenChange={setConfirmReplace}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>替换全部题目？</AlertDialogTitle>
            <AlertDialogDescription>
              将先清空现有的 {existingCount} 道题，再导入文件里的题目。校验失败时不会清空。此操作无法撤销。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction variant="destructive" onClick={() => onImport('replace')}>
              替换
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
