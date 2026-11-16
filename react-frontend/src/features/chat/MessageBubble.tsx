import type { MessageItem } from '../../types/api.ts'
import { AssistantMarkdown } from './AssistantMarkdown'

export function MessageBubble({ message }: { message: MessageItem }) {
  const isUser = message.role === 'user'
  const label = isUser ? 'You' : message.role === 'assistant' ? 'ResearchMate' : message.role

  return (
    <article aria-label={`${label} message`} className={`flex min-w-0 flex-col ${isUser ? 'items-end' : 'items-start'}`}>
      <p className="mb-2 px-1 text-xs font-medium text-muted">{label}</p>
      <div className={`min-w-0 max-w-full rounded-2xl border px-5 py-4 text-sm leading-7 ${isUser ? 'sm:max-w-[85%] border-accent/20 bg-accent/10 text-ink' : 'border-line bg-surface text-ink/90'}`}>
        {message.role === 'assistant' ? <AssistantMarkdown content={message.content} /> : <p className="whitespace-pre-wrap [overflow-wrap:anywhere]">{message.content}</p>}
      </div>
    </article>
  )
}
