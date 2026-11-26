import { useSession } from '../app/session/SessionContext'
import { LANGUAGE_CODES } from '../features/settings/model'
import { useChat } from '../features/chat/useChat'
import { ChatHistoryView } from '../features/chat/ChatHistoryView'
import { ChatInput } from '../features/chat/ChatInput'
import { useChatScroll } from '../features/chat/useChatScroll'
import { useOutletContext } from 'react-router'
import { lazy, Suspense } from 'react'
import type { ReactNode } from 'react'
import { useVoiceSession } from '../features/voice/useVoiceSession'

const VoiceMode = lazy(() => import('../features/voice/VoiceMode').then((module) => ({ default: module.VoiceMode })))

export function ChatPage() {
  const { chatHeader } = useOutletContext<{ chatHeader: ReactNode }>()
  const { session_id, settings, document_id, documentUpload, welcomeMessage } = useSession()
  const chat = useChat(session_id)
  const voice = useVoiceSession(session_id, chat.retry)
  const { viewportRef, contentRef, onScroll, followLatest } = useChatScroll(chat.messages)
  const uploadPending = documentUpload.status === 'uploading' || documentUpload.status === 'processing'
  const disabled = voice.busy || uploadPending || chat.status !== 'success' || chat.sending || chat.reconciling || Boolean(chat.syncError)

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div ref={viewportRef} onScroll={onScroll} tabIndex={0} role="region" aria-label="Chat messages" className="chat-scrollbar flex min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain focus-visible:outline-none focus-visible:shadow-[inset_0_0_0_1px_var(--color-line)] [overflow-anchor:none]">
        <div ref={contentRef} className="flex min-h-full shrink-0 flex-col">
          {chatHeader}
          {voice.busy ? <Suspense fallback={<section className="m-auto p-8 text-center"><p role="status" className="text-sm text-muted">Connecting?</p><button type="button" onClick={() => voice.finish()} className="mt-6 min-h-11 rounded-full border border-line px-5 text-sm">End conversation</button></section>}><VoiceMode room={voice.room} syncing={voice.phase === 'syncing' || (voice.phase === 'idle' && voice.busy)} onEnd={voice.finish} onMicrophoneError={voice.microphoneFailed} /></Suspense> : <ChatHistoryView history={chat} onRetry={chat.retry} welcomeMessage={welcomeMessage} languageCode={LANGUAGE_CODES[settings.language]} />}
          {voice.error && <p role="alert" className="mx-auto w-full max-w-3xl px-6 py-3 text-sm text-accent sm:px-10">{voice.error}</p>}
          {chat.sending && <p role="status" className="mx-auto w-full max-w-3xl px-6 pb-5 text-sm text-muted sm:px-10">Generating response…</p>}
          {chat.reconciling && <p role="status" className="mx-auto w-full max-w-3xl px-6 pb-5 text-sm text-muted sm:px-10">Syncing conversation history…</p>}
          {(chat.sendError || chat.syncError) && (
            <div className="mx-auto w-full max-w-3xl px-6 py-2 text-sm text-accent sm:px-10">
              <p role="alert">{chat.sendError} {chat.syncError}</p>
              {chat.syncError && <button type="button" disabled={voice.busy || chat.reconciling} onClick={() => { void chat.retry() }} className="mt-2 min-h-11 rounded-lg border border-accent/30 px-3 disabled:opacity-50">Check history</button>}
            </div>
          )}
        </div>
      </div>
      <ChatInput key={session_id} disabled={disabled} voiceActive={voice.busy} controlsDisabled={voice.busy || chat.sending || chat.reconciling} onStartVoice={() => { if (!disabled && !settings.agent_enabled && !document_id) void voice.start() }} onSend={(text) => {
        if (voice.busy || uploadPending || !chat.submit(text, settings, document_id)) return false
        followLatest()
        return true
      }} />
    </div>
  )
}
