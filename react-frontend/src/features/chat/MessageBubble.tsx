import type { MessageItem } from '../../types/api.ts'

export function MessageBubble({ message }: { message: MessageItem }) {
  const isUser = message.role === 'user'
  const label = isUser ? 'You' : message.role === 'assistant' ? 'ResearchMate' : message.role

  return (
    <article aria-label={`${label} message`} className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
      <p className="mb-2 px-1 text-xs font-medium text-muted">{label}</p>
      <div className={`max-w-full rounded-2xl border px-5 py-4 text-sm leading-7 sm:max-w-[85%] ${isUser ? 'border-accent/20 bg-accent/10 text-ink' : 'border-line bg-surface text-ink/90'}`}>
        <p className="whitespace-pre-wrap [overflow-wrap:anywhere]">{message.content}</p>
      </div>
    </article>
  )
}
