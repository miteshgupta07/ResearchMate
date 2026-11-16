import { ApiError, httpError } from './errors.ts'

export interface RequestOptions {
  signal?: AbortSignal
  /** Milliseconds, including response-body reading. Zero disables the timeout. */
  timeoutMs?: number
}

type Payload = { json?: unknown; formData?: never } | { json?: never; formData?: FormData }

export type HttpRequestOptions = RequestOptions & Payload & {
  method?: 'GET' | 'POST' | 'DELETE'
  query?: Record<string, string>
}

function resolveOrigin(value: string | undefined): string {
  try {
    const url = new URL(value?.trim() ?? '')
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password
      || url.pathname !== '/' || url.search || url.hash) throw new Error('Invalid origin')
    return url.origin
  } catch {
    throw new ApiError('configuration', 'Set VITE_API_BASE_URL to a valid HTTP(S) backend origin without a path, query, or credentials.')
  }
}

/** No requests occur until request() is called. No retries or credentials are added. */
export function createApiClient(baseUrl: string | (() => string | undefined)) {
  return {
    async request<T>(path: string, options: HttpRequestOptions = {}): Promise<T> {
      const timeoutMs = options.timeoutMs ?? 30_000
      if (!Number.isFinite(timeoutMs) || timeoutMs < 0 || timeoutMs > 2_147_483_647) {
        throw new ApiError('configuration', 'timeoutMs must be between 0 and 2147483647 milliseconds.')
      }
      const origin = resolveOrigin(typeof baseUrl === 'function' ? baseUrl() : baseUrl)
      // Only API-relative paths are accepted; never send a body to another origin.
      if (!path.startsWith('/') || path.startsWith('//') || path.includes('\\')) {
        throw new ApiError('configuration', 'API paths must start with a single slash.')
      }
      const url = new URL(path, origin)
      if (url.origin !== origin) {
        throw new ApiError('configuration', 'API paths must stay on the configured backend origin.')
      }
      for (const [key, value] of Object.entries(options.query ?? {})) url.searchParams.set(key, value)

      const headers = new Headers({ Accept: 'application/json' })
      let body: BodyInit | undefined = options.formData
      if (options.json !== undefined) {
        headers.set('Content-Type', 'application/json')
        body = JSON.stringify(options.json)
      }
      // The browser sets the multipart Content-Type and boundary for FormData.
      const controller = new AbortController()
      const cancel = () => controller.abort(new ApiError('cancelled', 'API request cancelled.', { cause: options.signal?.reason }))
      if (options.signal?.aborted) cancel()
      else options.signal?.addEventListener('abort', cancel, { once: true })

      const timer = timeoutMs > 0 ? setTimeout(() => {
        controller.abort(new ApiError('timeout', `API request timed out after ${timeoutMs} ms.`))
      }, timeoutMs) : undefined

      try {
        controller.signal.throwIfAborted()
        const response = await fetch(url, {
          method: options.method ?? 'GET', headers, body,
          signal: controller.signal, credentials: 'omit',
        })
        const text = await response.text()
        controller.signal.throwIfAborted()
        let payload: unknown
        try {
          payload = JSON.parse(text)
        } catch (cause) {
          // Preserve HTTP status even if a proxy/server returns HTML or plain text.
          if (!response.ok) throw httpError(response.status, text)
          throw new ApiError('response', 'API returned an empty or non-JSON response.', { status: response.status, cause })
        }
        if (!response.ok) throw httpError(response.status, payload)
        // DTOs provide compile-time contracts; JSON is not runtime schema-validated.
        return payload as T
      } catch (cause) {
        if (controller.signal.aborted) throw controller.signal.reason
        if (cause instanceof ApiError) throw cause
        throw new ApiError('network', 'Unable to reach the API or read its response. Check the connection and backend CORS configuration.', { cause })
      } finally {
        clearTimeout(timer)
        options.signal?.removeEventListener('abort', cancel)
      }
    },
  }
}

// Lazy configuration keeps the UI/build independent of backend availability.
export const apiClient = createApiClient(() => import.meta.env?.VITE_API_BASE_URL)
