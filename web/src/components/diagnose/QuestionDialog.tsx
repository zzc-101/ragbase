import { Loader2 } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import type { EvalQuestion, QuestionErrorItem, QuestionInput } from '@/lib/api'

interface QuestionDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  /** 编辑的题目；null 表示新增 */
  question: EvalQuestion | null
  pending: boolean
  /** 服务端校验（422）返回的逐条错误 */
  errors: QuestionErrorItem[]
  onSubmit: (input: QuestionInput) => void
}

const orNull = (value: string) => (value.trim() === '' ? null : value.trim())

export function QuestionDialog({ open, onOpenChange, question, pending, errors, onSubmit }: QuestionDialogProps) {
  return (
    <Dialog open={open} onOpenChange={(next) => !pending && onOpenChange(next)}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-lg">
        {/* 内容只在打开时挂载：每次打开都从题目当前值重新初始化表单 */}
        <QuestionForm question={question} pending={pending} errors={errors} onSubmit={onSubmit} onCancel={() => onOpenChange(false)} />
      </DialogContent>
    </Dialog>
  )
}

function QuestionForm({
  question,
  pending,
  errors,
  onSubmit,
  onCancel,
}: Pick<QuestionDialogProps, 'question' | 'pending' | 'errors' | 'onSubmit'> & { onCancel: () => void }) {
  const [text, setText] = useState(question?.question ?? '')
  const [reference, setReference] = useState(question?.reference ?? '')
  const [difficulty, setDifficulty] = useState(question?.difficulty ?? '')
  const [type, setType] = useState(question?.type ?? '')
  const [topic, setTopic] = useState(question?.source_topic ?? '')

  const submit = (event: FormEvent) => {
    event.preventDefault()
    onSubmit({
      question: text.trim(),
      reference: orNull(reference),
      difficulty: orNull(difficulty),
      type: orNull(type),
      source_topic: orNull(topic),
    })
  }

  const fieldErrors = (field: string) => errors.filter((item) => item.field === field)
  const otherErrors = errors.filter((item) => !['question', 'reference', 'difficulty', 'type', 'source_topic'].includes(item.field ?? ''))
  const hint = (field: string) =>
    fieldErrors(field).map((item) => (
      <p key={item.message} className="text-xs text-destructive">
        {item.message}
      </p>
    ))

  return (
    <form onSubmit={submit} className="grid gap-5">
      <DialogHeader>
        <DialogTitle>{question ? `编辑题目 #${question.id}` : '新增题目'}</DialogTitle>
        <DialogDescription>参考答案可以留空：这样的题只统计耗时，不统计召回率。</DialogDescription>
      </DialogHeader>

      <div className="grid gap-4">
        <label className="grid gap-1.5 text-sm">
          <span className="font-medium">问题</span>
          <Textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            required
            rows={3}
            maxLength={2000}
            aria-invalid={fieldErrors('question').length > 0 || undefined}
            placeholder="例如：什么是 TCP 三次握手？"
          />
          {hint('question')}
        </label>
        <label className="grid gap-1.5 text-sm">
          <span className="font-medium">参考答案（选填）</span>
          <Textarea
            value={reference}
            onChange={(event) => setReference(event.target.value)}
            rows={3}
            aria-invalid={fieldErrors('reference').length > 0 || undefined}
            placeholder="用来算词面召回率的要点，多个要点可用逗号或分号隔开"
          />
          {hint('reference')}
        </label>
        <div className="grid gap-4 sm:grid-cols-3">
          <label className="grid gap-1.5 text-sm">
            <span className="font-medium">难度（选填）</span>
            <Input value={difficulty} onChange={(event) => setDifficulty(event.target.value)} aria-invalid={fieldErrors('difficulty').length > 0 || undefined} />
            {hint('difficulty')}
          </label>
          <label className="grid gap-1.5 text-sm">
            <span className="font-medium">题型（选填）</span>
            <Input value={type} onChange={(event) => setType(event.target.value)} aria-invalid={fieldErrors('type').length > 0 || undefined} />
            {hint('type')}
          </label>
          <label className="grid gap-1.5 text-sm">
            <span className="font-medium">出处（选填）</span>
            <Input value={topic} onChange={(event) => setTopic(event.target.value)} aria-invalid={fieldErrors('source_topic').length > 0 || undefined} />
            {hint('source_topic')}
          </label>
        </div>
        {otherErrors.length > 0 && (
          <ul role="alert" className="space-y-0.5 rounded-lg bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {otherErrors.map((item) => (
              <li key={item.message}>{item.message}</li>
            ))}
          </ul>
        )}
      </div>

      <DialogFooter>
        <Button type="button" variant="outline" disabled={pending} onClick={onCancel}>
          取消
        </Button>
        <Button type="submit" disabled={pending || text.trim() === ''}>
          {pending && <Loader2 className="animate-spin" />}
          {question ? '保存' : '添加'}
        </Button>
      </DialogFooter>
    </form>
  )
}
