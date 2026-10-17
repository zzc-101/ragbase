import { BookOpen, MessageSquare } from 'lucide-react'
import { Link, Outlet, useLocation } from 'react-router'
import { useChat } from '@/hooks/useChat'
import { cn } from '@/lib/utils'
import { StatusPill } from './StatusPill'
import { ThemeToggle } from './ThemeToggle'

function Wordmark() {
  return (
    <div className="flex items-center gap-2.5">
      <span className="grid size-6 place-items-center rounded-md bg-primary" aria-hidden>
        <span className="grid gap-[3px]">
          <span className="block h-[2px] w-3 rounded-full bg-primary-foreground" />
          <span className="block h-[2px] w-3 rounded-full bg-primary-foreground/80" />
          <span className="block h-[2px] w-2 rounded-full bg-primary-foreground/60" />
        </span>
      </span>
      <span className="hidden text-[15px] font-semibold tracking-wide sm:inline">RAG 知识库</span>
    </div>
  )
}

const navClass = (isActive: boolean) =>
  cn(
    'inline-flex h-8 items-center gap-1.5 rounded-md px-2.5 text-sm sm:px-3 transition-colors focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none',
    isActive ? 'bg-accent font-medium text-foreground' : 'text-muted-foreground hover:bg-accent/60 hover:text-foreground',
  )

export function AppShell() {
  const { sessionId } = useChat()
  const { pathname } = useLocation()
  const chatHome = sessionId ? `/c/${sessionId}` : '/'
  const inLibrary = pathname.startsWith('/library')
  return (
    <div className="flex h-full flex-col">
      <header className="flex h-12 shrink-0 items-center justify-between gap-2 border-b bg-background px-3 sm:px-4">
        <div className="flex min-w-0 items-center gap-3 sm:gap-6">
          <Link to={chatHome} aria-label="RAG 知识库 首页" className="rounded-md focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none">
            <Wordmark />
          </Link>
          <nav className="flex items-center gap-1" aria-label="主导航">
            <Link to={chatHome} className={navClass(!inLibrary)} aria-current={inLibrary ? undefined : 'page'}>
              <MessageSquare className="size-4" />
              对话
            </Link>
            <Link to="/library" className={navClass(inLibrary)} aria-current={inLibrary ? 'page' : undefined}>
              <BookOpen className="size-4" />
              知识库
            </Link>
          </nav>
        </div>
        <div className="flex shrink-0 items-center gap-1 sm:gap-2">
          <StatusPill />
          <ThemeToggle />
        </div>
      </header>
      <main className="min-h-0 flex-1">
        <Outlet />
      </main>
    </div>
  )
}
