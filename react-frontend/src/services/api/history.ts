import type { HistoryClearResponse, HistoryResponse } from '../../types/api.ts'
import { apiClient } from './client.ts'
import type { RequestOptions } from './client.ts'

export function getChatHistory(sessionId: string, options: RequestOptions = {}): Promise<HistoryResponse> {
  return apiClient.request('/history', { ...options, query: { session_id: sessionId } })
}

export function clearChatHistory(sessionId: string, options: RequestOptions = {}): Promise<HistoryClearResponse> {
  return apiClient.request('/history', { ...options, method: 'DELETE', query: { session_id: sessionId } })
}
