import { DEFAULT_SETTINGS } from './model'
import type { ChatSettings } from './model'

export function ResetGenerationSettings({ onChange, descriptionId }: { onChange: (patch: Partial<ChatSettings>) => void; descriptionId: string }) {
  return <button type="button" onClick={() => onChange({ temperature: DEFAULT_SETTINGS.temperature, max_tokens: DEFAULT_SETTINGS.max_tokens })} aria-describedby={descriptionId} className="min-h-11 w-full rounded-xl border border-line bg-surface px-3 text-xs font-medium text-ink transition-colors hover:border-accent/40 hover:bg-accent/5 focus-visible:outline-accent">Reset to Defaults</button>
}
