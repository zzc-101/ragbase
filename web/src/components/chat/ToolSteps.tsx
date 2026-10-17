import { BookOpen, ChevronRight, Loader2, Search } from 'lucide-react'
import { useState } from 'react'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { describeStep, type ToolStep } from '@/lib/chat-types'
import { cn } from '@/lib/utils'

function StepIcon({ step }: { step: ToolStep }) {
  if (step.status === 'running') return <Loader2 className="size-3.5 animate-spin" />
  return step.name === 'list_documents' ? <BookOpen className="size-3.5" /> : <Search className="size-3.5" />
}

/** 助手回答上方的一行淡色步骤：折叠时显示最近一步，展开看全部。 */
export function ToolSteps({ steps }: { steps: ToolStep[] }) {
  const [open, setOpen] = useState(false)
  if (!steps.length) return null
  const last = steps[steps.length - 1]
  const more = steps.length - 1

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="mb-3 text-[13px] text-muted-foreground">
      <CollapsibleTrigger
        className="group flex max-w-full items-center gap-1.5 rounded-md py-0.5 text-left transition-colors hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
        aria-label={open ? '收起检索步骤' : '展开检索步骤'}
      >
        <StepIcon step={last} />
        <span className="truncate">{describeStep(last)}</span>
        {more > 0 && <span className="shrink-0 opacity-70">等 {steps.length} 步</span>}
        <ChevronRight className={cn('size-3.5 shrink-0 transition-transform duration-150', open && 'rotate-90')} />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <ul className="mt-1.5 space-y-1 border-l pl-3">
          {steps.map((step) => (
            <li key={step.id} className="flex items-start gap-1.5">
              <span className="mt-0.5">
                <StepIcon step={step} />
              </span>
              <span className="min-w-0 break-words">{describeStep(step)}</span>
            </li>
          ))}
        </ul>
      </CollapsibleContent>
    </Collapsible>
  )
}
