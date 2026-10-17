import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Loader2, RefreshCw } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { BookCard } from '@/components/library/BookCard'
import { JobCard } from '@/components/library/JobCard'
import { UploadDropzone } from '@/components/library/UploadDropzone'
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
import { queryKeys, useDocuments } from '@/hooks/queries'
import { useIngestJob } from '@/hooks/useIngestJob'
import { ApiError, deleteDocument, type DocumentInfo } from '@/lib/api'
import { displayBookName } from '@/lib/books'

export function LibraryPage() {
  const queryClient = useQueryClient()
  const documents = useDocuments()
  const ingest = useIngestJob()
  const [pendingDelete, setPendingDelete] = useState<DocumentInfo | null>(null)
  const [confirmRescan, setConfirmRescan] = useState(false)

  const remove = useMutation({
    mutationFn: (name: string) => deleteDocument(name),
    onSuccess: (result) => {
      toast.success(`已删除 ${displayBookName(result.name)}`, { description: `清除 ${result.chunks_deleted} 个片段` })
      void queryClient.invalidateQueries({ queryKey: queryKeys.documents })
      void queryClient.invalidateQueries({ queryKey: queryKeys.health })
    },
    onError: (error) => {
      if (error instanceof ApiError && error.status === 409) {
        toast.warning('已有入库任务在运行', { description: error.message })
      } else {
        toast.error('删除失败', { description: error instanceof Error ? error.message : undefined })
      }
    },
    onSettled: () => setPendingDelete(null),
  })

  const docs = documents.data
  const totalChunks = docs?.reduce((sum, doc) => sum + doc.chunk_count, 0) ?? 0

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto w-full max-w-5xl space-y-6 px-4 py-8 sm:px-6">
        <header className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">知识库</h1>
            <p className="tabular mt-1 text-sm text-muted-foreground">
              {docs ? (
                <>
                  共 <span className="text-foreground">{docs.length}</span> 本书，
                  <span className="text-foreground"> {totalChunks.toLocaleString()}</span> 个片段
                </>
              ) : documents.isError ? (
                '无法加载书目'
              ) : (
                '加载中…'
              )}
            </p>
          </div>
          <Button variant="outline" disabled={ingest.busy} onClick={() => setConfirmRescan(true)}>
            {ingest.rescan.isPending ? <Loader2 className="animate-spin" /> : <RefreshCw />}
            重新扫描
          </Button>
        </header>

        <UploadDropzone
          disabled={ingest.busy}
          uploading={ingest.upload.isPending}
          progress={ingest.uploadProgress}
          onFile={(file) => ingest.upload.mutate(file)}
        />

        {ingest.job && <JobCard job={ingest.job} onDismiss={ingest.dismiss} />}

        {documents.isLoading && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3" aria-hidden>
            {[0, 1, 2].map((i) => (
              <div key={i} className="h-32 animate-pulse rounded-xl bg-muted" />
            ))}
          </div>
        )}
        {documents.isError && (
          <div className="rounded-xl border border-dashed px-6 py-10 text-center text-sm text-muted-foreground">
            {documents.error instanceof ApiError && documents.error.status === 503
              ? '索引不存在或为空：请先运行 uv run ingest_incremental.py 建立索引，再重启服务。'
              : documents.error instanceof Error
                ? documents.error.message
                : '加载失败'}
            <div className="mt-3">
              <Button variant="outline" size="sm" onClick={() => void documents.refetch()}>
                重试
              </Button>
            </div>
          </div>
        )}
        {docs && docs.length === 0 && (
          <div className="rounded-xl border border-dashed px-6 py-10 text-center text-sm text-muted-foreground">
            还没有文档。上传一份 PDF / Word / Markdown，或把文件放进数据目录后点击“重新扫描”。
          </div>
        )}
        {docs && docs.length > 0 && (
          <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {docs.map((doc) => (
              <li key={doc.name}>
                <BookCard
                  doc={doc}
                  deleting={remove.isPending && remove.variables === doc.name}
                  onDelete={() => setPendingDelete(doc)}
                />
              </li>
            ))}
          </ul>
        )}
      </div>

      <AlertDialog open={confirmRescan} onOpenChange={setConfirmRescan}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>重新扫描数据目录？</AlertDialogTitle>
            <AlertDialogDescription>
              将对比数据目录与索引：新增或修改的文件重新入库，已删除的文件移出索引。
              如果索引是旧版本或入库规则有更新，会清空后全量重建，可能需要十几分钟，期间检索结果不完整。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction onClick={() => ingest.rescan.mutate()}>开始扫描</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={pendingDelete !== null} onOpenChange={(open) => !open && !remove.isPending && setPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>删除这本书？</AlertDialogTitle>
            <AlertDialogDescription>
              将删除《{pendingDelete && displayBookName(pendingDelete.name)}》的源文件和索引中的 {pendingDelete?.chunk_count} 个片段，
              并重建 BM25 索引（可能需要几十秒）。此操作无法撤销。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={remove.isPending}>取消</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={remove.isPending}
              onClick={(event) => {
                event.preventDefault() // 等删除完成后再关闭对话框，期间显示进度
                if (pendingDelete) remove.mutate(pendingDelete.name)
              }}
            >
              {remove.isPending && <Loader2 className="animate-spin" />}
              {remove.isPending ? '删除中…' : '删除'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </ScrollArea>
  )
}
