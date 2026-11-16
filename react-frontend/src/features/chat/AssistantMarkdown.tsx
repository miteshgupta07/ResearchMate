import Markdown, { defaultUrlTransform } from 'react-markdown'
import remarkGfm from 'remark-gfm'

function safeUrl(value: string): string {
  const safe = defaultUrlTransform(value)
  const protocol = /^([a-z][a-z0-9+.-]*):/i.exec(safe)?.[1]?.toLowerCase()
  return protocol && !['http', 'https', 'mailto'].includes(protocol) ? '' : safe
}

export function AssistantMarkdown({ content }: { content: string }) {
  return (
    <div className="assistant-markdown">
      <Markdown remarkPlugins={[remarkGfm]} skipHtml urlTransform={safeUrl} components={{
        a: ({ href, children, title }) => {
          if (!href) return <span>{children}</span>
          const external = /^(https?:)?\/\//i.test(href)
          return <a href={href} title={title} target={external ? '_blank' : undefined} rel={external ? 'noopener noreferrer' : undefined}>{children}</a>
        },
        table: ({ children }) => <div className="markdown-table-scroll chat-scrollbar" role="region" aria-label="Response table" tabIndex={0}><table>{children}</table></div>,
        pre: ({ children }) => <pre className="chat-scrollbar" tabIndex={0}>{children}</pre>,
        img: ({ alt }) => <span>{alt}</span>,
      }}>{content}</Markdown>
    </div>
  )
}
