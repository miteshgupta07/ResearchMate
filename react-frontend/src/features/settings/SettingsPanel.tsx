import { useId } from 'react'
import { useSession } from '../../app/session/SessionContext'
import { isLanguage, LANGUAGES } from './model'
import { Icon } from '../../components/ui/Icon'
import { SettingHelp } from './SettingHelp'
import { ResetGenerationSettings } from './ResetGenerationSettings'
import { SettingsDropdown } from '../../components/ui/SettingsDropdown'

export function SettingsPanel() {
  const id = useId()
  const { settings, updateSettings } = useSession()

  return (
    <details open className="mb-6 shrink-0 border-t border-line/70 px-3 pt-4 text-sm">
      <summary className="cursor-pointer rounded-sm py-2 font-medium text-ink/90"><span className="inline-flex items-center gap-2 align-middle">Chat settings<Icon name="settings" className="size-4 text-muted" /></span></summary>
      <div className="mt-4 space-y-5">
        <div>
          <div className="flex items-center justify-between gap-2"><label id={`${id}-language-label`} htmlFor={`${id}-language`} className="text-muted">Language</label><SettingHelp setting="Language">Choose the language for the welcome message and AI answers.</SettingHelp></div>
          <SettingsDropdown id={`${id}-language`} labelId={`${id}-language-label`} value={settings.language} options={LANGUAGES} onChange={(language) => { if (isLanguage(language)) updateSettings({ language }) }} />
        </div>
        <div>
          <div className="flex items-center justify-between gap-2"><label id={`${id}-model-label`} htmlFor={`${id}-model`} className="text-muted">Model</label><SettingHelp setting="Model">The AI model that answers your questions. GPT OSS is currently available.</SettingHelp></div>
          <SettingsDropdown id={`${id}-model`} labelId={`${id}-model-label`} descriptionId={`${id}-model-help`} value={settings.model_type} options={['GPT OSS'] as const} onChange={() => updateSettings({ model_type: 'GPT OSS' })} />
          <p id={`${id}-model-help`} className="mt-2 text-xs leading-5 text-muted">The available language model.</p>
        </div>
        <div>
          <div className="flex justify-between gap-2">
            <label htmlFor={`${id}-temperature`} className="text-muted">Temperature</label>
            <div className="flex items-center gap-1"><output htmlFor={`${id}-temperature`} className="tabular-nums text-ink">{settings.temperature.toFixed(2)}</output><SettingHelp setting="Temperature">Lower values give more focused answers. Higher values allow more creativity.</SettingHelp></div>
          </div>
          <input id={`${id}-temperature`} type="range" min="0" max="1" step="0.01" value={settings.temperature} onChange={(event) => updateSettings({ temperature: event.target.valueAsNumber })} aria-describedby={`${id}-temperature-help`} className="mt-1 min-h-11 w-full accent-accent" />
          <p id={`${id}-temperature-help`} className="text-xs leading-5 text-muted">Lower for focused answers, higher for creativity.</p>
        </div>
        <div>
          <div className="flex justify-between gap-2">
            <label htmlFor={`${id}-tokens`} className="text-muted">Max tokens</label>
            <div className="flex items-center gap-1"><output htmlFor={`${id}-tokens`} className="tabular-nums text-ink">{settings.max_tokens}</output><SettingHelp setting="Max tokens">Sets the response length limit. More tokens allow longer answers.</SettingHelp></div>
          </div>
          <input id={`${id}-tokens`} type="range" min="1" max="2048" step="1" value={settings.max_tokens} onChange={(event) => updateSettings({ max_tokens: event.target.valueAsNumber })} aria-describedby={`${id}-tokens-help`} className="mt-1 min-h-11 w-full accent-accent" />
          <p id={`${id}-tokens-help`} className="text-xs leading-5 text-muted">Controls the generated response length.</p>
        </div>
        <ResetGenerationSettings onChange={updateSettings} descriptionId={`${id}-reset-help`} />
        <p id={`${id}-reset-help`} className="sr-only">Resets Temperature and Max tokens only. Other settings and your attachment stay unchanged.</p>
      </div>
    </details>
  )
}
