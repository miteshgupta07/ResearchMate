/** Wire contracts from FastAPI schemas. Optional request fields may also be null. */
export interface GenerationOptions {
  language?: string | null
  model_type?: string | null
  /** FastAPI accepts 0 through 1 inclusive. */
  temperature?: number | null
  /** Positive integer; the backend schema has no upper bound. */
  max_tokens?: number | null
}

export interface ChatRequest extends GenerationOptions {
  session_id: string
  message: string
}

export interface MessageItem {
  role: string
  content: string
}

export type ChatResponse = MessageItem

export interface RAGQueryRequest extends ChatRequest {
  document_id: string
}

export interface RAGQueryResponse extends MessageItem {
  sources: unknown[]
}

export interface AgentRequest extends ChatRequest {
  document_id?: string | null
}

export interface AgentResponse extends MessageItem {
  intent: string
  confidence: number | null
  metadata: Record<string, unknown>
}

export interface DocumentUploadResponse {
  document_id: string
  filename: string
  status: string
}

export interface HistoryResponse {
  session_id: string
  messages: MessageItem[]
}

export interface HistoryClearResponse {
  status: string
}

export interface VoiceSessionRequest {
  session_id: string
}

export interface VoiceSessionResponse {
  url: string
  room_name: string
  token: string
}

export interface HealthResponse {
  status: string
}

export interface ValidationError {
  loc: (string | number)[]
  msg: string
  type: string
  input?: unknown
  ctx?: Record<string, unknown>
}
