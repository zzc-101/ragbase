import { memo, useMemo } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Citation } from '@/lib/api'
import { remarkCitations } from '@/lib/remark-citations'
import { cn } from '@/lib/utils'
import { CitationChip } from './CitationChip'

const REMARK_PLUGINS = [remarkGfm, remarkCitations]

interface Props {
  content: string
  citations: Citation[]
  streaming?: boolean
  onSelectCitation?: (n: number) => void
}

export const AnswerMarkdown = memo(function AnswerMarkdown({ content, citations, streaming, onSelectCitation }: Props) {
  const components = useMemo(() => {
    const byNumber = new Map(citations.map((citation) => [citation.n, citation]))
    return {
      'citation-chip': (props: { n?: string }) => {
        const n = Number(props.n)
        return <CitationChip n={n} citation={byNumber.get(n)} onSelect={onSelectCitation} />
      },
      a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
      table: ({ node: _node, ...props }) => (
        <div className="table-wrap">
          <table {...props} />
        </div>
      ),
    } as Components
  }, [citations, onSelectCitation])

  return (
    <div className={cn('answer', streaming && 'is-streaming')}>
      <ReactMarkdown remarkPlugins={REMARK_PLUGINS} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  )
})
