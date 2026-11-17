import type { DocumentUploadResponse } from '../../types/api.ts'
import { apiClient } from './client.ts'
import type { RequestOptions } from './client.ts'

export function uploadDocument(file: File, options: RequestOptions = {}): Promise<DocumentUploadResponse> {
  const formData = new FormData()
  formData.append('file', file, file.name)
  return apiClient.request('/documents/upload', { ...options, timeoutMs: options.timeoutMs ?? 120_000, method: 'POST', formData })
}
