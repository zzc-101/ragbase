import { useMutation, useQueryClient } from '@tanstack/react-query'
import { PanelLeftClose, PanelLeftOpen, Plus, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router'
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
import { ScrollArea } from '@/components/ui/scroll-area'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useSessions, queryKeys } from '@/hooks/queries'
import { useChat } from '@/hooks/useChat'
import { ApiError, deleteSession, type SessionSummary } from '@/lib/api'
import { relativeTime } from '@/lib/time'
import { cn } from '@/lib/utils'

const RAIL_MAX = 8 // 收起成图标栏时最多显示几个会话

interface Props {
  activeId: string | undefined
  /** 选中会话 / 新建对话后回调（窄屏下用来关闭抽屉） */
  onNavigate?: () => void
  /** 仅常驻侧栏（≥768px）传入：是否收起成图标栏，以及切换回调；抽屉里不传 */
  collapsed?: boolean
  onToggleCollapsed?: () => void
}

function IconTip({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent side="right">{label}</TooltipContent>
    </Tooltip>
  )
}

export function SessionSidebar({ activeId, onNavigate, collapsed = false, onToggleCollapsed }: Props) {
  const { data: sessions, isLoading, isError } = useSessions()
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const { forget } = useChat()
  const [pendingDelete, setPendingDelete] = useState<SessionSummary | null>(null)

  const remove = useMutation({
    mutationFn: async (id: string) => {
      try {
        await deleteSession(id)
      } catch (error) {
        if (!(error instanceof ApiError && error.status === 404)) throw error // 404 = 已经不在了，当作成功
      }
      return id
    },
    onSuccess: (id) => {
      forget(id)
      void queryClient.invalidateQueries({ queryKey: queryKeys.sessions })
      if (id === activeId) navigate('/')
      toast.success('已删除会话')
    },
    onError: (error) => toast.error('删除失败', { description: error instanceof Error ? error.message : undefined }),
    onSettled: () => setPendingDelete(null),
  })

  // 收起时的图标栏：优先最近的会话，当前会话不在其中则顶掉最后一个
  const railSessions = (() => {
    const all = sessions ?? []
    const first = all.slice(0, RAIL_MAX)
    const active = all.find((s) => s.session_id === activeId)
    return active && !first.includes(active) ? [...first.slice(0, RAIL_MAX - 1), active] : first
  })()

  const expanded = (
    <>
      <div className="flex items-center gap-1.5 p-3">
        <Button asChild variant="outline" className="min-w-0 flex-1 justify-start bg-card">
          <Link to="/" onClick={onNavigate}>
            <Plus />
            新对话
          </Link>
        </Button>
        {onToggleCollapsed && (
          <IconTip label="收起会话栏">
            <Button variant="ghost" size="icon" aria-label="收起会话栏" className="text-muted-foreground" onClick={onToggleCollapsed}>
              <PanelLeftClose />
            </Button>
          </IconTip>
        )}
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <nav aria-label="会话列表" className="px-2 pb-3">
          {isLoading && (
            <div className="space-y-2 px-2 pt-1" aria-hidden>
              {[0, 1, 2].map((i) => (
                <div key={i} className="h-10 animate-pulse rounded-md bg-muted" />
              ))}
            </div>
          )}
          {isError && <p className="px-3 py-6 text-center text-sm text-muted-foreground">无法加载会话列表</p>}
          {sessions && sessions.length === 0 && (
            <p className="px-3 py-6 text-center text-sm text-muted-foreground">还没有对话</p>
          )}
          <ul className="space-y-0.5">
            {sessions?.map((session) => {
              const active = session.session_id === activeId
              return (
                <li key={session.session_id} className="group relative">
                  <Link
                    to={`/c/${session.session_id}`}
                    onClick={onNavigate}
                    aria-current={active ? 'page' : undefined}
                    className={cn(
                      'block rounded-lg px-3 py-2 pr-9 transition-colors duration-150 focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none',
                      active ? 'bg-accent' : 'hover:bg-accent/60',
                    )}
                  >
                    <span className="block truncate text-sm">{session.title}</span>
                    <span className="block truncate text-xs text-muted-foreground">{relativeTime(session.updated_at)}</span>
                  </Link>
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    aria-label={`删除会话：${session.title}`}
                    onClick={() => setPendingDelete(session)}
                    className="absolute top-1/2 right-1.5 -translate-y-1/2 text-muted-foreground opacity-0 transition-opacity duration-150 group-focus-within:opacity-100 group-hover:opacity-100 hover:text-destructive focus-visible:opacity-100"
                  >
                    <Trash2 />
                  </Button>
                </li>
              )
            })}
          </ul>
        </nav>
      </ScrollArea>
    </>
  )

  const rail = (
    <div className="flex h-full min-h-0 flex-col items-center gap-2 py-3">
      <IconTip label="展开会话栏">
        <Button variant="ghost" size="icon" aria-label="展开会话栏" className="text-muted-foreground" onClick={onToggleCollapsed}>
          <PanelLeftOpen />
        </Button>
      </IconTip>
      <IconTip label="新对话">
        <Button asChild variant="outline" size="icon" className="bg-card">
          <Link to="/" aria-label="新对话">
            <Plus />
          </Link>
        </Button>
      </IconTip>
      <nav aria-label="最近会话" className="mt-1 min-h-0 w-full flex-1 overflow-y-auto">
        <ul className="flex flex-col items-center gap-1.5 pb-1">
          {railSessions.map((session) => {
            const active = session.session_id === activeId
            return (
              <li key={session.session_id}>
                <IconTip label={session.title}>
                  <Link
                    to={`/c/${session.session_id}`}
                    aria-label={session.title}
                    aria-current={active ? 'page' : undefined}
                    className={cn(
                      'grid size-9 place-items-center rounded-lg border text-sm font-medium transition-colors duration-150 focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none',
                      active
                        ? 'border-primary/40 bg-primary/10 text-primary'
                        : 'border-transparent text-muted-foreground hover:bg-accent/60 hover:text-foreground',
                    )}
                  >
                    {Array.from(session.title.trim())[0] ?? '#'}
                  </Link>
                </IconTip>
              </li>
            )
          })}
        </ul>
      </nav>
    </div>
  )

  const dialog = (
    <AlertDialog open={pendingDelete !== null} onOpenChange={(open) => !open && setPendingDelete(null)}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>删除这个会话？</AlertDialogTitle>
          <AlertDialogDescription>「{pendingDelete?.title}」的对话记录会被清空，无法恢复。</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>取消</AlertDialogCancel>
          <AlertDialogAction variant="destructive" onClick={() => pendingDelete && remove.mutate(pendingDelete.session_id)}>
            删除
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )

  // 抽屉：只有完整列表
  if (!onToggleCollapsed) {
    return (
      <div className="flex h-full min-h-0 flex-col bg-sidebar">
        {expanded}
        {dialog}
      </div>
    )
  }

  // 常驻侧栏：完整列表与图标栏叠放，外层宽度过渡时两者交叉淡入淡出，内容本身宽度固定，不会被挤压变形
  return (
    <div className="relative h-full min-h-0 overflow-hidden bg-sidebar">
      <div
        inert={collapsed}
        className={cn(
          'absolute inset-y-0 left-0 flex w-[220px] flex-col transition-opacity duration-200 lg:w-[260px]',
          collapsed && 'pointer-events-none opacity-0',
        )}
      >
        {expanded}
      </div>
      <div
        inert={!collapsed}
        className={cn('absolute inset-y-0 left-0 w-14 transition-opacity duration-200', !collapsed && 'pointer-events-none opacity-0')}
      >
        {rail}
      </div>
      {dialog}
    </div>
  )
}
