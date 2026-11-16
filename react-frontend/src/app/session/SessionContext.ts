import { createContext, useContext } from 'react'
import type { ChatMode, ChatSettings } from '../../features/settings/model.ts'
import type { BrowserSession } from './storage.ts'

export interface SessionContextValue extends BrowserSession {
  document_id: string | null
  setDocumentId: (documentId: string | null) => void
  updateSettings: (settings: Partial<ChatSettings>) => void
  mode: ChatMode
  welcomeMessage: string
}

export const SessionContext = createContext<SessionContextValue | null>(null)

export function useSession(): SessionContextValue {
  const session = useContext(SessionContext)
  if (!session) throw new Error('useSession must be used within SessionProvider.')
  return session
}
