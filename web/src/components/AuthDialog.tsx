import { useQueryClient } from '@tanstack/react-query'
import { Eye, EyeOff, KeyRound, Loader2 } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { useAuth } from '@/hooks/useAuth'
import { connect } from '@/lib/auth'

/** 表单放在 DialogContent 里：对话框关闭时整体卸载，下次打开输入框是空的。 */
function KeyForm({ message }: { message: string | null }) {
  const queryClient = useQueryClient()
  const [value, setValue] = useState('')
  const [visible, setVisible] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const shown = error ?? message

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (busy) return
    setBusy(true)
    setError(null)
    const failure = await connect(value)
    setBusy(false)
    if (failure) {
      setError(failure)
      return
    }
    void queryClient.invalidateQueries() // 密钥生效：所有数据重新拉取（对话区的输入内容不受影响）
  }

  return (
    <form onSubmit={submit} className="grid gap-4">
      <div className="grid gap-2">
        <div className="relative">
          <Input
            type={visible ? 'text' : 'password'}
            value={value}
            onChange={(event) => {
              setValue(event.target.value)
              setError(null)
            }}
            placeholder="访问密钥"
            aria-label="访问密钥"
            aria-invalid={shown ? true : undefined}
            aria-describedby="auth-hint"
            autoComplete="off"
            autoCapitalize="off"
            spellCheck={false}
            className="pr-10"
          />
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            onClick={() => setVisible((v) => !v)}
            aria-label={visible ? '隐藏密钥' : '显示密钥'}
            aria-pressed={visible}
            className="absolute top-1/2 right-1.5 -translate-y-1/2"
          >
            {visible ? <EyeOff /> : <Eye />}
          </Button>
        </div>
        {shown && (
          <p role="alert" className="text-xs text-destructive">
            {shown}
          </p>
        )}
        <p id="auth-hint" className="text-xs text-muted-foreground">
          密钥由服务管理员通过 RAG_API_KEY 配置
        </p>
      </div>
      <Button type="submit" disabled={busy || !value.trim()}>
        {busy && <Loader2 className="animate-spin" />}
        连接
      </Button>
    </form>
  )
}

/** 需要访问密钥时的居中对话框：不可通过 Esc / 点击外部关闭，只能输入有效密钥后继续（Radix Dialog 自带焦点陷阱）。 */
export function AuthDialog() {
  const { dialogOpen, message } = useAuth()
  return (
    <Dialog open={dialogOpen}>
      <DialogContent
        showCloseButton={false}
        className="sm:max-w-sm"
        onEscapeKeyDown={(event) => event.preventDefault()}
        onPointerDownOutside={(event) => event.preventDefault()}
        onInteractOutside={(event) => event.preventDefault()}
      >
        <DialogHeader>
          <span className="mb-1 grid size-9 place-items-center rounded-full bg-accent text-muted-foreground" aria-hidden>
            <KeyRound className="size-4" />
          </span>
          <DialogTitle className="text-base">需要访问密钥</DialogTitle>
          <DialogDescription>这个服务启用了访问保护，请输入密钥后继续。</DialogDescription>
        </DialogHeader>
        <KeyForm message={message} />
      </DialogContent>
    </Dialog>
  )
}
