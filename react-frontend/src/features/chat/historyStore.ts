import { getChatHistory } from '../../services/api/history.ts'
import { ApiError } from '../../services/api/errors.ts'
import type { RequestOptions } from '../../services/api/client.ts'
import type { HistoryResponse, MessageItem } from '../../types/api.ts'

/** Local identity for a persisted snapshot; future pending messages stay separate. */
export interface ChatMessage extends MessageItem {
  id: string
  source: 'history' | 'local'
}

export interface HistoryState {
  status: 'loading' | 'success' | 'error'
  messages: ChatMessage[]
  error: string | null
}

type HistoryLoader = (sessionId: string, options: (RequestOptions & { waitForIdle?: boolean })) => Promise<HistoryResponse>

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.kind === 'configuration') return 'Chat history is unavailable. Set VITE_API_BASE_URL to your backend origin.'
    if (error.kind === 'timeout') return 'Loading chat history timed out. Please try again.'
    if (error.kind === 'network') return 'Unable to connect to chat history. Check your connection and try again.'
  }
  return 'Unable to load chat history. Please try again.'
}

function isHistory(value: unknown, sessionId: string): value is HistoryResponse {
  if (typeof value !== 'object' || value === null) return false
  const data = value as Partial<HistoryResponse>
  return data.session_id === sessionId && Array.isArray(data.messages)
    && data.messages.every((message: unknown) => {
      if (typeof message !== 'object' || message === null) return false
      const item = message as Partial<MessageItem>
      return typeof item.role === 'string' && typeof item.content === 'string'
    })
}

/** One hook instance owns one store; no global cache or browser persistence. */
export function createHistoryStore(sessionId: string, loader: HistoryLoader = getChatHistory) {
  let state: HistoryState = { status: 'loading', messages: [], error: null }
  const listeners = new Set<() => void>()
  let generation = 0
  let activeController: AbortController | undefined

  const publish = (next: HistoryState) => {
    state = next
    listeners.forEach((listener) => listener())
  }

  const cancel = () => {
    generation += 1
    activeController?.abort()
    activeController = undefined
  }

  return {
    getSnapshot: () => state,
    subscribe: (listener: () => void) => {
      listeners.add(listener)
      return () => { listeners.delete(listener) }
    },
    cancel,
    async load(options: { waitForIdle?: boolean } = {}): Promise<void> {
      cancel()
      const requestGeneration = generation
      const controller = new AbortController()
      activeController = controller
      const isCurrent = () => generation === requestGeneration && !controller.signal.aborted
      publish({ status: 'loading', messages: [], error: null })
      try {
        const result = await loader(sessionId, { signal: controller.signal, ...options })
        if (!isCurrent()) return
        if (!isHistory(result, sessionId)) throw new Error('Invalid history response')
        publish({
          status: 'success', error: null,
          messages: result.messages.map((message, index) => ({
            id: `${sessionId}:history:${index}`, source: 'history',
            role: message.role, content: message.content,
          })),
        })
      } catch (error) {
        if (isCurrent()) publish({ status: 'error', messages: [], error: errorMessage(error) })
      } finally {
        if (isCurrent()) activeController = undefined
      }
    },
  }
}
