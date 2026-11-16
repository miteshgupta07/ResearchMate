import { useSession } from '../../app/session/SessionContext'

export function AgentModeToggle() {
  const { settings, updateSettings } = useSession()
  return (
    <button type="button" role="switch" aria-label="Agent Mode" aria-checked={settings.agent_enabled} onClick={() => updateSettings({ agent_enabled: !settings.agent_enabled })} title={settings.agent_enabled ? 'Disable Agent Mode' : 'Enable Agent Mode'} className={`inline-flex min-h-11 items-center gap-2 rounded-full border px-3 text-xs font-medium transition-colors focus-visible:bg-accent/20 ${settings.agent_enabled ? 'border-accent/30 bg-accent/10 text-accent' : 'border-transparent text-muted hover:bg-canvas hover:text-ink'}`}>
      <span aria-hidden="true" className={`size-1.5 rounded-full ${settings.agent_enabled ? 'bg-accent' : 'bg-muted/50'}`} />
      Agent <span className="sr-only">{settings.agent_enabled ? 'enabled' : 'disabled'}</span>
    </button>
  )
}
