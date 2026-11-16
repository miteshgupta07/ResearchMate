import { useCallback, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { getChatMode, normalizeSettings, WELCOME_MESSAGES } from '../../features/settings/model.ts'
import type { ChatSettings } from '../../features/settings/model.ts'
import { SessionContext } from './SessionContext.ts'
import { getSessionStorage, loadSession, saveSession } from './storage.ts'

export function SessionProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState(() => loadSession(getSessionStorage()))
  // Future uploads set this only after successful processing. Never persist it.
  const [documentId, setDocumentId] = useState<string | null>(null)

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
    setDocumentId,
    updateSettings,
    mode: getChatMode(session.settings.agent_enabled, documentId),
    welcomeMessage: WELCOME_MESSAGES[session.settings.language],
  }), [session, documentId, updateSettings])

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
}
