import { AlertTriangle, Library, Loader2, PanelLeft } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router'
import { Composer } from '@/components/chat/Composer'
import { EmptyState } from '@/components/chat/EmptyState'
import { MessageList } from '@/components/chat/MessageList'
import { SessionSidebar } from '@/components/chat/SessionSidebar'
import { SourcesPanel } from '@/components/chat/SourcesPanel'
import { Button } from '@/components/ui/button'
import { Sheet, SheetContent, SheetDescription, SheetTitle } from '@/components/ui/sheet'
import { useHealth, useSessions } from '@/hooks/queries'
import { useChat } from '@/hooks/useChat'
import { useMediaQuery } from '@/hooks/useMediaQuery'
import { safeStorage } from '@/lib/storage'
import { cn } from '@/lib/utils'

const SIDEBAR_COLLAPSED_KEY = 'rag.sidebar.collapsed'

export function ChatPage() {
  const { sessionId: routeId } = useParams()
  const navigate = useNavigate()
  const chat = useChat()
  const { data: sessions } = useSessions()
  const { data: health } = useHealth()

  // ≥768px 会话栏常驻（可收起成图标栏）；<768px 改抽屉。依据面板 ≥1280px 才停靠，让位给会话导航。
  const dockedSidebar = useMediaQuery('(min-width: 768px)')
  const wideSources = useMediaQuery('(min-width: 1280px)')
  const [sidebarSheet, setSidebarSheet] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => safeStorage.get('local', SIDEBAR_COLLAPSED_KEY) === '1')
  const toggleSidebar = useCallback(() => {
    setSidebarCollapsed((prev) => {
      safeStorage.set('local', SIDEBAR_COLLAPSED_KEY, prev ? null : '1')
      return !prev
    })
  }, [])
  const [panelOpen, setPanelOpen] = useState(true) // 宽屏：右侧面板是否展开
  const [sourcesSheet, setSourcesSheet] = useState(false) // 窄屏：依据抽屉
  // 依据面板里手动选中的回答 / 编号；key 变化（换会话、来了新的带引用回答）后自动作废
  const [selection, setSelection] = useState<{ key: string; messageId: string; n: number | null } | null>(null)

  const { syncRoute } = chat
  useEffect(() => {
    syncRoute(routeId)
  }, [routeId, syncRoute])

  // 依据面板默认展示最近一条带引用的回答；用户点过某条回答的编号 / “依据”按钮后以那条为准
  const lastCited = useMemo(
    () => [...chat.messages].reverse().find((m) => m.role === 'assistant' && m.citations.length > 0),
    [chat.messages],
  )
  const selectionKey = `${routeId ?? ''}|${lastCited?.id ?? ''}`
  const activeSelection = selection?.key === selectionKey ? selection : null
  const shown = chat.messages.find((m) => m.id === activeSelection?.messageId) ?? lastCited
  const selectedN = activeSelection?.n ?? null

  const openSources = useCallback(
    (messageId: string, n: number | null) => {
      setSelection({ key: selectionKey, messageId, n })
      if (wideSources) setPanelOpen(true)
      else setSourcesSheet(true)
    },
    [wideSources, selectionKey],
  )
  const onSelectCitation = useCallback((messageId: string, n: number) => openSources(messageId, n), [openSources])
  const onOpenSources = useCallback((messageId: string) => openSources(messageId, null), [openSources])

  const title = sessions?.find((s) => s.session_id === chat.sessionId)?.title
  const heading = title ?? (chat.sessionId ? '对话' : '新对话')
  const empty = chat.messages.length === 0
  const agentDown = health && !(health.status === 'ok' && health.agent_ready)

  const panel = (onClose?: () => void) => (
    <SourcesPanel
      citations={shown?.citations ?? []}
      selectedN={selectedN}
      hasAnswer={shown !== undefined}
      onClose={onClose}
    />
  )

  return (
    <div className="flex h-full min-h-0">
      {dockedSidebar && (
        <aside
          className={cn(
            'shrink-0 border-r transition-[width] duration-200 ease-out',
            sidebarCollapsed ? 'w-14' : 'w-[220px] lg:w-[260px]',
          )}
          aria-label="会话"
        >
          <SessionSidebar activeId={chat.sessionId} collapsed={sidebarCollapsed} onToggleCollapsed={toggleSidebar} />
        </aside>
      )}

      <section className="flex min-w-0 flex-1 flex-col">
        <div className="flex h-11 shrink-0 items-center justify-between gap-2 border-b px-3">
          <div className="flex min-w-0 items-center gap-1">
            {dockedSidebar ? (
              <h2 className="truncate px-1 text-sm text-muted-foreground">{heading}</h2>
            ) : (
              <Button
                variant="ghost"
                size="sm"
                aria-label={`打开会话列表（当前：${heading}）`}
                className="-ml-1 min-w-0 justify-start gap-1.5 px-2 text-muted-foreground"
                onClick={() => setSidebarSheet(true)}
              >
                <PanelLeft />
                <span className="truncate">{heading}</span>
              </Button>
            )}
          </div>
          <Button
            variant="ghost"
            size="sm"
            aria-label={wideSources && panelOpen ? '收起依据面板' : '查看依据'}
            aria-pressed={wideSources ? panelOpen : sourcesSheet}
            className="text-muted-foreground"
            onClick={() => (wideSources ? setPanelOpen(!panelOpen) : setSourcesSheet(true))}
          >
            <Library />
            依据
            {(shown?.citations.length ?? 0) > 0 && <span className="tabular">{shown?.citations.length}</span>}
          </Button>
        </div>

        {agentDown && (
          <div role="status" className="flex items-center gap-2 border-b bg-warning/10 px-4 py-2 text-sm text-warning">
            <AlertTriangle className="size-4 shrink-0" />
            {health?.status === 'ok' ? '大模型未配置，暂时不能对话（仍可在知识库页管理文档）。' : '索引尚未就绪，暂时不能对话。'}
          </div>
        )}

        {chat.historyLoading ? (
          <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
            <Loader2 className="mr-2 size-4 animate-spin" />
            加载会话…
          </div>
        ) : chat.historyError ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
            <p className="max-w-sm text-sm text-muted-foreground">{chat.historyError}</p>
            <Button variant="outline" size="sm" onClick={() => navigate('/')}>
              开始新对话
            </Button>
          </div>
        ) : empty ? (
          <EmptyState onAsk={(q) => void chat.send(q)} disabled={chat.streaming} />
        ) : (
          <MessageList
            messages={chat.messages}
            streaming={chat.streaming}
            onSelectCitation={onSelectCitation}
            onOpenSources={onOpenSources}
            onRetry={chat.retry}
          />
        )}

        <Composer
          streaming={chat.streaming}
          sources={chat.selectedSources}
          onSourcesChange={chat.setSelectedSources}
          onSend={(text) => void chat.send(text)}
          onStop={chat.stop}
        />
      </section>

      {wideSources && panelOpen && (
        <aside className="w-[340px] shrink-0 border-l" aria-label="依据">
          {panel(() => setPanelOpen(false))}
        </aside>
      )}

      {!dockedSidebar && (
        <Sheet open={sidebarSheet} onOpenChange={setSidebarSheet}>
          <SheetContent side="left" className="w-[min(86vw,300px)] p-0" showCloseButton={false}>
            <SheetTitle className="sr-only">会话列表</SheetTitle>
            <SheetDescription className="sr-only">选择或删除历史会话，或开始新对话</SheetDescription>
            <SessionSidebar activeId={chat.sessionId} onNavigate={() => setSidebarSheet(false)} />
          </SheetContent>
        </Sheet>
      )}

      {!wideSources && (
        <Sheet open={sourcesSheet} onOpenChange={setSourcesSheet}>
          <SheetContent side="right" className="w-[min(92vw,380px)] p-0" showCloseButton={false}>
            <SheetTitle className="sr-only">依据</SheetTitle>
            <SheetDescription className="sr-only">本条回答引用的教材片段</SheetDescription>
            {panel(() => setSourcesSheet(false))}
          </SheetContent>
        </Sheet>
      )}
    </div>
  )
}
