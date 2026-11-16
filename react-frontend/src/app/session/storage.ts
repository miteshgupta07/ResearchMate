import { normalizeSettings } from '../../features/settings/model.ts'
import type { ChatSettings } from '../../features/settings/model.ts'

export const SESSION_STORAGE_KEY = 'researchmate.session.v1'

export interface BrowserSession {
  session_id: string
  settings: ChatSettings
}

const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

export function getSessionStorage(): Storage | null {
  try {
    return window.sessionStorage
  } catch {
    return null
  }
}

export function saveSession(storage: Storage | null, session: BrowserSession): void {
  try {
    // Explicit allowlist: no documents, messages, files, credentials, or arbitrary keys.
    storage?.setItem(SESSION_STORAGE_KEY, JSON.stringify({
      session_id: session.session_id, settings: normalizeSettings(session.settings),
    }))
  } catch {
    // Storage may be disabled or full; the current React session stays usable.
  }
}

export function loadSession(storage: Storage | null, createId = () => crypto.randomUUID()): BrowserSession {
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
    session_id: typeof data.session_id === 'string' && UUID_V4.test(data.session_id) ? data.session_id : createId(),
    settings: normalizeSettings(data.settings),
  }
  // Save the UUID immediately so StrictMode's repeated initializer reuses it.
  saveSession(storage, session)
  return session
}
