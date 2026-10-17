import { FileUp, Loader2 } from 'lucide-react'
import { useRef, useState } from 'react'
import { toast } from 'sonner'
import { Progress } from '@/components/ui/progress'
import { cn } from '@/lib/utils'

export const ACCEPTED_EXTENSIONS = ['.pdf', '.docx', '.md', '.markdown', '.txt']

/** 客户端预检：返回错误文案，通过则返回 null。大小上限由服务端（RAG_MAX_UPLOAD_MB）判定，超限会收到 413。 */
export function validateFile(file: File): string | null {
  const name = file.name
  const dot = name.lastIndexOf('.')
  const extension = dot >= 0 ? name.slice(dot).toLowerCase() : ''
  if (extension === '.doc') return '不支持旧版 .doc：请先在 Word 里另存为 .docx 再上传'
  if (!ACCEPTED_EXTENSIONS.includes(extension)) {
    return `不支持的格式 ${extension || '（无扩展名）'}；支持：${ACCEPTED_EXTENSIONS.join(' ')}`
  }
  if (file.size === 0) return '文件是空的'
  return null
}

interface Props {
  disabled: boolean
  uploading: boolean
  progress: number | null
  onFile: (file: File) => void
}

export function UploadDropzone({ disabled, uploading, progress, onFile }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  const handle = (files: FileList | null) => {
    if (!files || files.length === 0) return
    if (files.length > 1) toast.info('一次只能上传一个文件', { description: `已选第一个：${files[0].name}` })
    const file = files[0]
    const problem = validateFile(file)
    if (problem) {
      toast.error('无法上传', { description: problem })
      return
    }
    onFile(file)
  }

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault()
        if (!disabled) setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault()
        setDragging(false)
        if (!disabled) handle(event.dataTransfer.files)
      }}
      className={cn(
        'relative rounded-xl border border-dashed bg-card/60 transition-colors duration-150',
        dragging && 'border-primary bg-primary/5',
        disabled && 'opacity-60',
      )}
    >
      <button
        type="button"
        disabled={disabled}
        onClick={() => input.current?.click()}
        className="flex w-full flex-col items-center gap-1.5 rounded-xl px-6 py-7 text-center focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
      >
        {uploading ? <Loader2 className="size-5 animate-spin text-primary" /> : <FileUp className="size-5 text-primary" />}
        <span className="text-sm font-medium">
          {uploading ? '上传中…' : dragging ? '松开以上传' : '拖拽文件到这里，或点击选择'}
        </span>
        <span className="text-xs text-muted-foreground">支持 PDF、Word（.docx）、Markdown、TXT；同名文件会覆盖</span>
      </button>
      {uploading && progress !== null && (
        <Progress
          value={Math.round(progress * 100)}
          className="absolute inset-x-4 bottom-2 w-auto"
          aria-label="上传进度"
        />
      )}
      <input
        ref={input}
        type="file"
        hidden
        accept={ACCEPTED_EXTENSIONS.join(',')}
        onChange={(event) => {
          handle(event.target.files)
          event.target.value = '' // 允许再次选择同一个文件
        }}
      />
    </div>
  )
}
