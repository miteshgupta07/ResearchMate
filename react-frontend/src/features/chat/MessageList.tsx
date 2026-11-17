import type { ChatMessage } from './historyStore.ts'
import { MessageBubble } from './MessageBubble.tsx'

export function MessageList({ messages }: { messages: readonly ChatMessage[] }) {
  return (
    <ol aria-label="Conversation history" className="space-y-6">
      {messages.map((message) => <li key={message.id}><MessageBubble message={message} /></li>)}
    </ol>
  )
}
