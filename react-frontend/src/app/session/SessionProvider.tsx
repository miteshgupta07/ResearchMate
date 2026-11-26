import { useCallback, useEffect, useMemo, useState, useSyncExternalStore } from 'react'
import type { ReactNode } from 'react'
import { getChatMode, normalizeSettings, WELCOME_MESSAGES } from '../../features/settings/model.ts'
import type { ChatSettings } from '../../features/settings/model.ts'
import { SessionContext } from './SessionContext.ts'
import { getSessionStorage, loadSession, saveSession } from './storage.ts'
import { createDocumentStore } from '../../features/documents/documentStore.ts'

export function SessionProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState(() => loadSession(getSessionStorage()))
  const [documents] = useState(() => createDocumentStore())
  // Survives chat-page navigation, without persisting tokens or voice state.
  const [voiceHistoryPending, setVoiceHistoryPending] = useState(false)
  const documentUpload = useSyncExternalStore(documents.subscribe, documents.getSnapshot, documents.getSnapshot)
  const documentId = documentUpload.document?.document_id ?? null

  useEffect(() => documents.remove, [documents, session.session_id])

  useEffect(() => {
    saveSession(getSessionStorage(), session)
  }, [session])

  const updateSettings = useCallback((patch: Partial<ChatSettings>) => {
    setSession((previous) => ({
      ...previous, settings: normalizeSettings({ ...previous.settings, ...patch }),
    }))
  }, [])

  const value = useMemo(() => ({
    ...session,
    document_id: documentId,
    setDocumentId: documents.setDocumentId,
    documentUpload,
    selectDocument: documents.select,
    removeDocument: documents.remove,
    updateSettings,
    mode: getChatMode(session.settings.agent_enabled, documentId),
    welcomeMessage: WELCOME_MESSAGES[session.settings.language],
    voiceHistoryPending,
    setVoiceHistoryPending,
  }), [session, documentId, documentUpload, documents, updateSettings, voiceHistoryPending])

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
}
