import { Navigate, Route, Routes } from 'react-router'
import { AppShell } from '@/components/AppShell'
import { Toaster } from '@/components/ui/sonner'
import { TooltipProvider } from '@/components/ui/tooltip'
import { ChatProvider } from '@/hooks/useChat'
import { ChatPage } from '@/pages/ChatPage'
import { LibraryPage } from '@/pages/LibraryPage'

export default function App() {
  return (
    <TooltipProvider delayDuration={300}>
      <ChatProvider>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<ChatPage />} />
            <Route path="c/:sessionId" element={<ChatPage />} />
            <Route path="library" element={<LibraryPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </ChatProvider>
      <Toaster position="top-center" />
    </TooltipProvider>
  )
}
