import type { HealthResponse } from '../../types/api.ts'
import { apiClient } from './client.ts'
import type { RequestOptions } from './client.ts'

export function checkBackendHealth(options: RequestOptions = {}): Promise<HealthResponse> {
  return apiClient.request('/health', { ...options, timeoutMs: options.timeoutMs ?? 5_000 })
}
