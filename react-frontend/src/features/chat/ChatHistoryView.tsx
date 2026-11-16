import type { HistoryState } from './historyStore.ts'
import { MessageList } from './MessageList.tsx'
import { PagePlaceholder } from '../../components/ui/PagePlaceholder.tsx'

interface ChatHistoryViewProps {
  history: HistoryState
  onRetry: () => void
  welcomeMessage: string
  languageCode: string
}

export function ChatHistoryView({ history, onRetry, welcomeMessage, languageCode }: ChatHistoryViewProps) {
  if (history.status === 'success' && history.messages.length === 0) {
    return <PagePlaceholder icon="spark" eyebrow="YOUR RESEARCH WORKSPACE" title="Chat" description={welcomeMessage} descriptionLang={languageCode} note="Ask a question to start your conversation." />
  }

  if (history.status === 'loading' || history.status === 'error') {
    return (
      <section aria-labelledby="chat-heading" className="mx-auto flex w-full max-w-3xl flex-1 flex-col items-center justify-center px-6 py-16 text-center">
        <h1 id="chat-heading" className="text-4xl font-semibold tracking-tight sm:text-5xl">Chat</h1>
        {history.status === 'loading' ? (
          <p role="status" className="mt-6 flex items-center gap-3 text-sm text-muted">
            <span aria-hidden="true" className="size-4 animate-spin rounded-full border-2 border-line border-t-accent motion-reduce:animate-none" />
            Loading conversation history…
          </p>
        ) : (
          <>
            <p role="alert" className="mt-6 max-w-md text-sm leading-7 text-muted">{history.error}</p>
            <button type="button" onClick={onRetry} className="mt-6 min-h-11 rounded-xl border border-accent/30 bg-accent/10 px-5 text-sm font-medium text-accent transition-colors hover:bg-accent/20">Retry history</button>
          </>
        )}
      </section>
    )
  }

  return (
    <section aria-labelledby="chat-heading" className="mx-auto w-full max-w-3xl px-6 py-8 sm:px-10">
      <h1 id="chat-heading" className="mb-8 text-2xl font-semibold tracking-tight">Chat</h1>
      <MessageList messages={history.messages} />
    </section>
  )
}
