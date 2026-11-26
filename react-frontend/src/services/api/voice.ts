import type { VoiceSessionRequest, VoiceSessionResponse } from '../../types/api.ts'
import { apiClient } from './client.ts'
import type { RequestOptions } from './client.ts'
import { ApiError } from './errors.ts'

export async function createVoiceSession(request: VoiceSessionRequest, options: RequestOptions = {}): Promise<VoiceSessionResponse> {
  const result = await apiClient.request<VoiceSessionResponse>('/voice/session', { ...options, method: 'POST', json: request })
  try {
    const url = new URL(result.url)
    if (url.protocol !== 'wss:' || url.username || url.password || !result.token?.trim() || !result.room_name?.trim()) throw new Error()
  } catch {
    throw new ApiError('response', 'Voice connection details are unavailable.')
  }
  return result
}
