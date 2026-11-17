import { normalizeSettings } from '../../features/settings/model.ts'
import type { ChatSettings } from '../../features/settings/model.ts'

export const SESSION_STORAGE_KEY = 'researchmate.session.v1'

export interface BrowserSession {
  session_id: string
  settings: ChatSettings
}

/** One closure per loaded JS page; no storage restoration, including in StrictMode. */
export function createPageSessionId(createId = () => crypto.randomUUID()): () => string {
  let id: string | undefined
  return () => { id ??= createId(); return id }
}

const pageSessionId = createPageSessionId()

export function getSessionStorage(): Storage | null {
  try {
    return window.sessionStorage
  } catch {
    return null
  }
}

export function saveSession(storage: Storage | null, session: BrowserSession): void {
  try {
    // Only settings survive refresh. IDs, documents and messages stay in memory.
    storage?.setItem(SESSION_STORAGE_KEY, JSON.stringify({
      settings: normalizeSettings(session.settings),
    }))
  } catch {
    // Storage may be disabled or full; the current React session stays usable.
  }
}

export function loadSession(storage: Storage | null, createId = pageSessionId): BrowserSession {
  let data: Record<string, unknown> = {}
  try {
    const parsed: unknown = JSON.parse(storage?.getItem(SESSION_STORAGE_KEY) ?? 'null')
    if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
      data = parsed as Record<string, unknown>
    }
  } catch {
    // Invalid JSON and blocked storage both fall back to a fresh session.
  }
  const session = {
    session_id: createId(),
    settings: normalizeSettings(data.settings),
  }
  // Rewrite legacy snapshots without the old session ID while retaining settings.
  saveSession(storage, session)
  return session
}
