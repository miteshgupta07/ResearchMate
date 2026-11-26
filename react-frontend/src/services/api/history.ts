import type { HistoryClearResponse, HistoryResponse } from '../../types/api.ts'
import { apiClient } from './client.ts'
import type { RequestOptions } from './client.ts'

export function getChatHistory(sessionId: string, options: RequestOptions & { waitForIdle?: boolean } = {}): Promise<HistoryResponse> {
  return apiClient.request('/history', { ...options, timeoutMs: options.waitForIdle ? 65_000 : options.timeoutMs, query: { session_id: sessionId, ...(options.waitForIdle ? { wait_for_idle: 'true' } : {}) } })
}

export function clearChatHistory(sessionId: string, options: RequestOptions = {}): Promise<HistoryClearResponse> {
  return apiClient.request('/history', { ...options, method: 'DELETE', query: { session_id: sessionId } })
}
