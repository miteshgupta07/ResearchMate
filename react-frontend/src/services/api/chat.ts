import type { AgentRequest, AgentResponse, ChatRequest, ChatResponse, RAGQueryRequest, RAGQueryResponse } from '../../types/api.ts'
import { apiClient } from './client.ts'
import type { RequestOptions } from './client.ts'

export function sendChatMessage(request: ChatRequest, options: RequestOptions = {}): Promise<ChatResponse> {
  return apiClient.request('/chat', { ...options, timeoutMs: options.timeoutMs ?? 120_000, method: 'POST', json: request })
}

export function sendRagQuery(request: RAGQueryRequest, options: RequestOptions = {}): Promise<RAGQueryResponse> {
  return apiClient.request('/rag/query', { ...options, timeoutMs: options.timeoutMs ?? 120_000, method: 'POST', json: request })
}

export function sendAgentMessage(request: AgentRequest, options: RequestOptions = {}): Promise<AgentResponse> {
  return apiClient.request('/agent/route', { ...options, timeoutMs: options.timeoutMs ?? 120_000, method: 'POST', json: request })
}
