import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Plus, Trash2 } from 'lucide-react'
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
import { useSessions, queryKeys } from '@/hooks/queries'
import { useChat } from '@/hooks/useChat'
import { ApiError, deleteSession, type SessionSummary } from '@/lib/api'
import { relativeTime } from '@/lib/time'
import { cn } from '@/lib/utils'

interface Props {
  activeId: string | undefined
  /** 选中会话 / 新建对话后回调（窄屏下用来关闭抽屉） */
  onNavigate?: () => void
}

export function SessionSidebar({ activeId, onNavigate }: Props) {
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

  return (
    <div className="flex h-full min-h-0 flex-col bg-sidebar">
      <div className="p-3">
        <Button asChild variant="outline" className="w-full justify-start bg-card">
          <Link to="/" onClick={onNavigate}>
            <Plus />
            新对话
          </Link>
        </Button>
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
                    <span className="block text-xs text-muted-foreground">{relativeTime(session.updated_at)}</span>
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

      <AlertDialog open={pendingDelete !== null} onOpenChange={(open) => !open && setPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>删除这个会话？</AlertDialogTitle>
            <AlertDialogDescription>
              「{pendingDelete?.title}」的对话记录会被清空，无法恢复。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => pendingDelete && remove.mutate(pendingDelete.session_id)}
            >
              删除
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
