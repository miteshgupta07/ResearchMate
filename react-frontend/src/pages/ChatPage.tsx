import { useSession } from '../app/session/SessionContext'
import { LANGUAGE_CODES } from '../features/settings/model'
import { useChat } from '../features/chat/useChat'
import { ChatHistoryView } from '../features/chat/ChatHistoryView'
import { ChatInput } from '../features/chat/ChatInput'
import { useChatScroll } from '../features/chat/useChatScroll'
import { PdfUpload } from '../features/documents/PdfUpload'

export function ChatPage() {
  const { session_id, settings, document_id, documentUpload, welcomeMessage } = useSession()
  const chat = useChat(session_id)
  const { viewportRef, contentRef, onScroll, followLatest } = useChatScroll(chat.messages)
  const supportedMode = !settings.agent_enabled
  const uploadPending = documentUpload.status === 'uploading' || documentUpload.status === 'processing'
  const disabled = !supportedMode || uploadPending || chat.status !== 'success' || chat.sending || chat.reconciling || Boolean(chat.syncError)

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
      {!supportedMode && <p className="px-6 pt-3 text-center text-xs text-muted">Turn off Agent mode to use normal chat or document Q&A. Agent execution is coming later.</p>}
      <PdfUpload disabled={chat.sending || chat.reconciling} />
      <ChatInput key={session_id} disabled={disabled} onSend={(text) => {
        if (!supportedMode || uploadPending || !chat.submit(text, settings, document_id)) return false
        followLatest()
        return true
      }} />
    </div>
  )
}
