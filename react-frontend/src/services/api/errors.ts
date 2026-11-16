import type { ValidationError } from '../../types/api.ts'

export type ApiErrorKind = 'configuration' | 'http' | 'validation' | 'network' | 'response' | 'timeout' | 'cancelled'

interface ApiErrorOptions extends ErrorOptions {
  status?: number
  details?: unknown
  validationErrors?: ValidationError[]
}

export class ApiError extends Error {
  readonly kind: ApiErrorKind
  readonly status?: number
  readonly details: unknown
  readonly validationErrors: ValidationError[]

  constructor(kind: ApiErrorKind, message: string, options: ApiErrorOptions = {}) {
    super(message, { cause: options.cause })
    this.name = 'ApiError'
    this.kind = kind
    this.status = options.status
    this.details = options.details
    this.validationErrors = options.validationErrors ?? []
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isValidationError(value: unknown): value is ValidationError {
  return isRecord(value) && Array.isArray(value.loc)
    && value.loc.every((part) => typeof part === 'string' || typeof part === 'number')
    && typeof value.msg === 'string' && typeof value.type === 'string'
}

export function httpError(status: number, payload: unknown): ApiError {
  const detail = isRecord(payload) ? payload.detail : undefined
  const validationErrors = Array.isArray(detail) ? detail.filter(isValidationError) : []
  const message = typeof detail === 'string' && detail.trim()
    ? detail
    : validationErrors.length
      ? validationErrors.map((error) => `${error.loc.join('.')}: ${error.msg}`).join('; ')
      : `API request failed (HTTP ${status}).`

  return new ApiError(status === 422 ? 'validation' : 'http', message, {
    status, details: payload, validationErrors,
  })
}
