import { useSession } from '../app/session/SessionContext'
import { LANGUAGE_CODES } from '../features/settings/model'
import { useChat } from '../features/chat/useChat'
import { ChatHistoryView } from '../features/chat/ChatHistoryView'
import { ChatInput } from '../features/chat/ChatInput'
import { useChatScroll } from '../features/chat/useChatScroll'

export function ChatPage() {
  const { session_id, settings, document_id, welcomeMessage } = useSession()
  const chat = useChat(session_id)
  const { viewportRef, contentRef, onScroll, followLatest } = useChatScroll(chat.messages)
  const normalMode = !settings.agent_enabled && !document_id
  const disabled = !normalMode || chat.status !== 'success' || chat.sending || chat.reconciling || Boolean(chat.syncError)

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div ref={viewportRef} onScroll={onScroll} className="flex min-h-0 flex-1 flex-col overflow-y-auto">
        <div ref={contentRef} className="flex flex-1 flex-col">
          <ChatHistoryView history={chat} onRetry={chat.retry} welcomeMessage={welcomeMessage} languageCode={LANGUAGE_CODES[settings.language]} />
          {chat.sending && <p role="status" className="mx-auto w-full max-w-3xl px-6 pb-5 text-sm text-muted sm:px-10">Generating response…</p>}
          {chat.reconciling && <p role="status" className="mx-auto w-full max-w-3xl px-6 pb-5 text-sm text-muted sm:px-10">Syncing conversation history…</p>}
        </div>
      </div>
      {(chat.sendError || chat.syncError) && (
        <div className="mx-auto w-full max-w-3xl px-6 py-2 text-sm text-accent sm:px-10">
          <p role="alert">{chat.sendError} {chat.syncError}</p>
          {chat.syncError && <button type="button" disabled={chat.reconciling} onClick={() => { void chat.retry() }} className="mt-2 min-h-11 rounded-lg border border-accent/30 px-3 disabled:opacity-50">Check history</button>}
        </div>
      )}
      {!normalMode && <p className="px-6 pt-3 text-center text-xs text-muted">Normal chat is available with Agent mode off and no document attached. Other chat modes are coming later.</p>}
      <ChatInput key={session_id} disabled={disabled} onSend={(text) => {
        if (!normalMode || !chat.submit(text, settings)) return false
        followLatest()
        return true
      }} />
    </div>
  )
}
