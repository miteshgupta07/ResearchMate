import { createHistoryStore } from './historyStore.ts'
import type { ChatMessage, HistoryState } from './historyStore.ts'
import { sendAgentMessage, sendChatMessage, sendRagQuery } from '../../services/api/chat.ts'
import { ApiError } from '../../services/api/errors.ts'
import type { AgentRequest, AgentResponse, ChatRequest, ChatResponse, GenerationOptions, MessageItem, RAGQueryRequest, RAGQueryResponse } from '../../types/api.ts'
import type { RequestOptions } from '../../services/api/client.ts'

export interface ChatState extends HistoryState {
  sending: boolean
  reconciling: boolean
  sendError: string | null
  syncError: string | null
}

type Sender = (request: ChatRequest, options: RequestOptions) => Promise<ChatResponse>
type RagSender = (request: RAGQueryRequest, options: RequestOptions) => Promise<RAGQueryResponse>
type AgentSender = (request: AgentRequest, options: RequestOptions) => Promise<AgentResponse>
type HistoryStore = ReturnType<typeof createHistoryStore>

function sendErrorMessage(error: unknown, documentQuery: boolean): string {
  if (error instanceof ApiError) {
    if (documentQuery && (error.status === 404 || error.validationErrors.some((issue) => issue.loc.includes('document_id')))) {
      return 'This document is unavailable or invalid. Remove it or upload the PDF again before asking another document question.'
    }
    if (error.status === 409) return 'This conversation is busy. End Voice Mode or wait for the current response, then try again.'
    if (error.kind === 'timeout') return 'The response timed out. Your message may have been saved.'
    if (error.kind === 'network') return 'The connection failed. Your message may have been saved.'
    if (error.kind === 'configuration') return 'Chat is unavailable. Check VITE_API_BASE_URL.'
    if (error.kind === 'validation') return 'The message or settings were rejected. Check your settings before sending again.'
  }
  return 'The response could not be generated. Your message may have been saved.'
}

const sameMessage = (a: MessageItem, b: MessageItem) => a.role === b.role && a.content === b.content

/** Serializes POST + history reconciliation. No automatic retries of a POST. */
export function createChatStore(sessionId: string, history: HistoryStore = createHistoryStore(sessionId), sender: Sender = sendChatMessage, ragSender: RagSender = sendRagQuery, agentSender: AgentSender = sendAgentMessage) {
  let state: ChatState = { ...history.getSnapshot(), sending: false, reconciling: false, sendError: null, syncError: null }
  let confirmed: ChatMessage[] = []
  let local: ChatMessage[] = []
  let postSucceeded = false
  let persistence: 'normal' | 'local' | 'summary' = 'normal'
  const localTurns: { after: number; messages: ChatMessage[] }[] = []
  const summaryPrompts = new Map<number, { saved: MessageItem; original: string }>()
  let generation = 0
  let sequence = 0
  let busy = false
  let controller: AbortController | undefined
  const listeners = new Set<() => void>()
  const publish = (patch: Partial<ChatState>) => {
    state = { ...state, ...patch }
    listeners.forEach((listener) => listener())
  }

  // Anchors count server messages only; local agent turns never shift reconciliation offsets.
  const displayMessages = (saved: ChatMessage[]): ChatMessage[] => {
    const display: ChatMessage[] = []
    for (let index = 0; index <= saved.length; index += 1) {
      for (const turn of localTurns) if (Math.min(turn.after, saved.length) === index) display.push(...turn.messages)
      if (index < saved.length) {
        const override = summaryPrompts.get(index)
        display.push(override && sameMessage(override.saved, saved[index])
          ? { ...saved[index], content: override.original } : saved[index])
      }
    }
    return display
  }

  const cancel = () => {
    generation += 1
    controller?.abort()
    controller = undefined
    history.cancel()
    busy = false
  }

  const sync = async (token: number, initial: boolean, options: { waitForIdle?: boolean } = {}) => {
    await history.load(options)
    if (generation !== token) return
    const loaded = history.getSnapshot()
    if (loaded.status !== 'success') {
      if (initial) publish({ ...loaded })
      else publish({ syncError: 'History could not be refreshed. Check history before sending another message.' })
      return
    }
    if (postSucceeded && local.length) {
      // Compare only the new turn after the known baseline. Never deduplicate by text globally.
      const baselineMatches = loaded.messages.length >= confirmed.length
        && confirmed.every((message, index) => sameMessage(message, loaded.messages[index]))
      let matched = 0
      if (baselineMatches) {
        while (matched < local.length && loaded.messages[confirmed.length + matched]
          && (sameMessage(local[matched], loaded.messages[confirmed.length + matched])
            || (persistence === 'summary' && matched === 0 && loaded.messages[confirmed.length].role === 'user'))) matched += 1
      }
      const localOnly = persistence === 'local' && baselineMatches && matched === 0
      if (!baselineMatches || (matched < local.length && !localOnly)) {
        // Preserve the visible successful response until the server confirms the entire turn.
        publish({ syncError: 'The latest response is not yet confirmed in history. Check history before sending another message.' })
        return
      }
      if (localOnly) localTurns.push({ after: confirmed.length, messages: local })
      if (persistence === 'summary' && !localOnly) {
        summaryPrompts.set(confirmed.length, { saved: loaded.messages[confirmed.length], original: local[0].content })
      }
    }
    confirmed = loaded.messages
    local = []
    postSucceeded = false
    publish({ ...loaded, messages: displayMessages(confirmed), syncError: null })
  }

  const load = async (options: { waitForIdle?: boolean } = {}) => {
    if (busy) return
    busy = true
    const token = ++generation
    const initial = state.status !== 'success'
    if (initial) publish({ status: 'loading', error: null, messages: [], sending: false })
    else publish({ sending: false, reconciling: true })
    try {
      await sync(token, initial, options)
    } finally {
      if (generation === token) {
        busy = false
        publish({ reconciling: false })
      }
    }
  }

  const submit = (text: string, settings: GenerationOptions & { agent_enabled?: boolean }, documentId: string | null = null): boolean => {
    const content = text.trim()
    if (!content || busy || state.status !== 'success' || state.syncError) return false
    if (documentId !== null && (typeof documentId !== 'string' || !documentId.trim())) {
      publish({ sendError: 'The document ID is missing or invalid. Remove the attachment or upload the PDF again.' })
      return false
    }
    busy = true
    const token = ++generation
    controller = new AbortController()
    const signal = controller.signal
    local = [{ id: `${sessionId}:local:${++sequence}`, source: 'local', role: 'user', content }]
    postSucceeded = false
    persistence = 'normal'
    const agentEnabled = Boolean(settings.agent_enabled)
    publish({ messages: [...displayMessages(confirmed), ...local], sending: true, sendError: null, syncError: null })
    // Capture settings and document selection for this turn; never reroute in flight.
    const request: ChatRequest = {
      session_id: sessionId, message: content, language: settings.language,
      model_type: settings.model_type, temperature: settings.temperature, max_tokens: settings.max_tokens,
    }
    void (async () => {
      try {
        const response = agentEnabled
          ? await agentSender({ ...request, document_id: documentId }, { signal })
          : documentId === null ? await sender(request, { signal })
            : await ragSender({ ...request, document_id: documentId }, { signal })
        if (generation !== token) return
        if (response?.role !== 'assistant' || typeof response.content !== 'string') throw new Error('Invalid chat response')
        if (agentEnabled) {
          const intent = (response as AgentResponse).intent
          if (typeof intent !== 'string') throw new Error('Invalid agent response')
          persistence = intent === 'INTENT_ARXIV_SEARCH' || intent === 'INTENT_ARXIV_RECOMMEND'
            ? 'local' : intent === 'INTENT_RAG_SUMMARY' ? 'summary' : 'normal'
        }
        local = [...local, { id: `${sessionId}:local:${++sequence}`, source: 'local', role: 'assistant', content: response.content }]
        postSucceeded = true
        publish({ messages: [...displayMessages(confirmed), ...local] })
      } catch (error) {
        if (generation !== token) return
        publish({ sendError: sendErrorMessage(error, documentId !== null) })
      }
      if (generation !== token) return
      publish({ sending: false, reconciling: true })
      try {
        await sync(token, false)
      } finally {
        if (generation === token) {
          controller = undefined
          busy = false
          publish({ sending: false, reconciling: false })
        }
      }
    })()
    return true
  }

  return {
    getSnapshot: () => state,
    subscribe: (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener) } },
    load, submit, cancel,
  }
}
