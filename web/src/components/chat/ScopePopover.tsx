import { BookMarked } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useDocuments } from '@/hooks/queries'
import { displayBookName } from '@/lib/books'

interface Props {
  selected: string[]
  onChange: (sources: string[]) => void
}

/** “范围”：限定只在哪几本书里检索（作为 /chat/stream 的 sources）。不选 = 全部。 */
export function ScopePopover({ selected, onChange }: Props) {
  const { data: documents, isLoading, isError } = useDocuments()

  const toggle = (name: string, checked: boolean) =>
    onChange(checked ? [...selected, name] : selected.filter((item) => item !== name))

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="sm" className="text-muted-foreground" aria-label="选择检索范围">
          <BookMarked />
          范围
          {selected.length > 0 && (
            <Badge className="tabular h-4 min-w-4 px-1 text-[10px]" aria-label={`已选 ${selected.length} 本`}>
              {selected.length}
            </Badge>
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" side="top" className="w-80 gap-3 p-3">
        <div className="flex items-center justify-between">
          <p className="text-sm font-medium">检索范围</p>
          <Button variant="ghost" size="xs" disabled={!selected.length} onClick={() => onChange([])}>
            清除
          </Button>
        </div>
        <p className="-mt-1 text-xs text-muted-foreground">
          {selected.length ? `只在选中的 ${selected.length} 本书里检索` : '未选择：在全部书目里检索'}
        </p>
        {isLoading && <p className="py-4 text-center text-sm text-muted-foreground">加载书目…</p>}
        {isError && <p className="py-4 text-center text-sm text-muted-foreground">无法加载书目</p>}
        {documents && documents.length === 0 && (
          <p className="py-4 text-center text-sm text-muted-foreground">知识库还没有文档</p>
        )}
        {documents && documents.length > 0 && (
          <ScrollArea className="-mx-1 max-h-64">
            <ul className="px-1">
              {documents.map((doc) => {
                const id = `scope-${doc.name}`
                return (
                  <li key={doc.name}>
                    <label
                      htmlFor={id}
                      className="flex cursor-pointer items-start gap-2.5 rounded-md px-2 py-2 text-sm transition-colors hover:bg-accent"
                    >
                      <Checkbox
                        id={id}
                        className="mt-0.5"
                        checked={selected.includes(doc.name)}
                        onCheckedChange={(checked) => toggle(doc.name, checked === true)}
                      />
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span className="min-w-0 flex-1 leading-snug">{displayBookName(doc.name)}</span>
                        </TooltipTrigger>
                        <TooltipContent side="right" className="max-w-sm break-all">
                          {doc.name}
                        </TooltipContent>
                      </Tooltip>
                      <span className="tabular shrink-0 text-xs text-muted-foreground">{doc.chunk_count}</span>
                    </label>
                  </li>
                )
              })}
            </ul>
          </ScrollArea>
        )}
      </PopoverContent>
    </Popover>
  )
}
