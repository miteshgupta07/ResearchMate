export const LANGUAGES = ['English', 'Hindi', 'Spanish', 'French', 'German'] as const
export type Language = typeof LANGUAGES[number]

export interface ChatSettings {
  language: Language
  model_type: 'GPT OSS'
  temperature: number
  max_tokens: number
  agent_enabled: boolean
}

export const DEFAULT_SETTINGS: Readonly<ChatSettings> = {
  language: 'English', model_type: 'GPT OSS', temperature: 0.7,
  max_tokens: 512, agent_enabled: false,
}

export const WELCOME_MESSAGES: Record<Language, string> = {
  English: 'Hi! How can I assist you today?',
  Hindi: 'नमस्ते! मैं आज आपकी कैसे मदद कर सकता हूँ?',
  Spanish: '¡Hola! ¿Cómo puedo ayudarte hoy?',
  French: "Bonjour! Comment puis-je vous aider aujourd'hui?",
  German: 'Hallo! Wie kann ich Ihnen heute helfen?',
}

export const LANGUAGE_CODES: Record<Language, string> = {
  English: 'en', Hindi: 'hi', Spanish: 'es', French: 'fr', German: 'de',
}

export type ChatMode = 'Agent Mode' | 'Document Q&A' | 'Normal Chat'

export function getChatMode(agentEnabled: boolean, documentId: string | null): ChatMode {
  return agentEnabled ? 'Agent Mode' : documentId ? 'Document Q&A' : 'Normal Chat'
}

export function isLanguage(value: unknown): value is Language {
  return LANGUAGES.some((language) => language === value)
}

/** Validate restored or programmatic values; ignore unknown keys entirely. */
export function normalizeSettings(value: unknown): ChatSettings {
  const data = typeof value === 'object' && value !== null ? value as Record<string, unknown> : {}
  return {
    language: isLanguage(data.language) ? data.language : DEFAULT_SETTINGS.language,
    model_type: 'GPT OSS',
    temperature: typeof data.temperature === 'number' && Number.isFinite(data.temperature)
      && data.temperature >= 0 && data.temperature <= 1 ? data.temperature : DEFAULT_SETTINGS.temperature,
    max_tokens: typeof data.max_tokens === 'number' && Number.isInteger(data.max_tokens)
      && data.max_tokens >= 1 && data.max_tokens <= 2048 ? data.max_tokens : DEFAULT_SETTINGS.max_tokens,
    agent_enabled: typeof data.agent_enabled === 'boolean' ? data.agent_enabled : DEFAULT_SETTINGS.agent_enabled,
  }
}
