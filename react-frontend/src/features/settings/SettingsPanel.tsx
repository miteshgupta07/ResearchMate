import { useId } from 'react'
import { useSession } from '../../app/session/SessionContext'
import { isLanguage, LANGUAGES } from './model'

const selectClass = 'mt-2 min-h-11 w-full rounded-lg border border-line bg-surface px-3 text-sm text-ink'

export function SettingsPanel() {
  const id = useId()
  const { settings, updateSettings } = useSession()

  return (
    <details open className="mb-6 shrink-0 border-t border-line/70 px-3 pt-4 text-sm">
      <summary className="cursor-pointer rounded-sm py-2 font-medium text-ink/90">Chat settings</summary>
      <div className="mt-4 space-y-5">
        <div>
          <label className="flex min-h-11 cursor-pointer items-center justify-between gap-3" htmlFor={`${id}-agent`}>
            <span>Agent mode</span>
            <input id={`${id}-agent`} type="checkbox" role="switch" checked={settings.agent_enabled} onChange={(event) => updateSettings({ agent_enabled: event.target.checked })} aria-describedby={`${id}-agent-help`} className="size-5 shrink-0 accent-accent" />
          </label>
          <p id={`${id}-agent-help`} className="text-xs leading-5 text-muted">{settings.agent_enabled ? 'Advanced research mode selected.' : 'Use normal chat or document Q&A.'}</p>
        </div>
        <div>
          <label htmlFor={`${id}-language`} className="text-muted">Language</label>
          <select id={`${id}-language`} value={settings.language} onChange={(event) => { if (isLanguage(event.target.value)) updateSettings({ language: event.target.value }) }} className={selectClass}>
            {LANGUAGES.map((language) => <option key={language} value={language}>{language}</option>)}
          </select>
        </div>
        <div>
          <label htmlFor={`${id}-model`} className="text-muted">Model</label>
          <select id={`${id}-model`} value={settings.model_type} onChange={() => updateSettings({ model_type: 'GPT OSS' })} className={selectClass} aria-describedby={`${id}-model-help`}>
            <option value="GPT OSS">GPT OSS</option>
          </select>
          <p id={`${id}-model-help`} className="mt-2 text-xs leading-5 text-muted">The available language model.</p>
        </div>
        <div>
          <div className="flex justify-between gap-2">
            <label htmlFor={`${id}-temperature`} className="text-muted">Temperature</label>
            <output htmlFor={`${id}-temperature`} className="tabular-nums text-ink">{settings.temperature.toFixed(2)}</output>
          </div>
          <input id={`${id}-temperature`} type="range" min="0" max="1" step="0.01" value={settings.temperature} onChange={(event) => updateSettings({ temperature: event.target.valueAsNumber })} aria-describedby={`${id}-temperature-help`} className="mt-1 min-h-11 w-full accent-accent" />
          <p id={`${id}-temperature-help`} className="text-xs leading-5 text-muted">Lower for focused answers, higher for creativity.</p>
        </div>
        <div>
          <div className="flex justify-between gap-2">
            <label htmlFor={`${id}-tokens`} className="text-muted">Max tokens</label>
            <output htmlFor={`${id}-tokens`} className="tabular-nums text-ink">{settings.max_tokens}</output>
          </div>
          <input id={`${id}-tokens`} type="range" min="1" max="2048" step="1" value={settings.max_tokens} onChange={(event) => updateSettings({ max_tokens: event.target.valueAsNumber })} aria-describedby={`${id}-tokens-help`} className="mt-1 min-h-11 w-full accent-accent" />
          <p id={`${id}-tokens-help`} className="text-xs leading-5 text-muted">Controls the generated response length.</p>
        </div>
      </div>
    </details>
  )
}
