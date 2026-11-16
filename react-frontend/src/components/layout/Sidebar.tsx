import { Link, NavLink } from 'react-router'
import { navigation } from '../../app/navigation'
import { Icon } from '../ui/Icon'
import { SettingsPanel } from '../../features/settings/SettingsPanel'

type SidebarProps = {
  collapsed?: boolean
  onNavigate?: () => void
  onToggle: () => void
  mobile?: boolean
}

export function Sidebar({ collapsed = false, onNavigate, onToggle, mobile = false }: SidebarProps) {
  const toggleLabel = mobile ? 'Close navigation' : collapsed ? 'Expand sidebar' : 'Collapse sidebar'

  return (
    <div className="flex h-full flex-col overflow-y-auto p-3">
      <div className={`flex min-h-16 shrink-0 items-center gap-2 ${collapsed ? 'flex-col py-2' : 'justify-between px-2'}`}>
        <Link to="/" onClick={onNavigate} aria-label="ResearchMate home" className="flex items-center gap-2.5 rounded-lg">
          <span className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-accent/10 text-accent"><Icon name="spark" className="size-6" /></span>
          {!collapsed && <span className="text-[17px] font-semibold tracking-tight">ResearchMate<span className="text-accent">.</span></span>}
        </Link>
        <button type="button" onClick={onToggle} aria-label={toggleLabel} title={toggleLabel} aria-expanded={mobile ? undefined : !collapsed} aria-controls={mobile ? undefined : 'desktop-navigation'} className="flex size-10 shrink-0 items-center justify-center rounded-lg text-muted transition-colors hover:bg-surface hover:text-ink">
          <Icon name={mobile ? 'close' : 'panel'} />
        </button>
      </div>

      <nav id={mobile ? 'mobile-navigation' : 'desktop-navigation'} aria-label="Main navigation" className="mt-8 mb-8 shrink-0">
        <p className={`mb-3 px-3 text-[10px] font-semibold tracking-[0.18em] text-muted ${collapsed ? 'sr-only' : ''}`}>WORKSPACE</p>
        <ul className="space-y-1.5">
          {navigation.map(({ path, label, icon }) => (
            <li key={path}>
              <NavLink to={path} end onClick={onNavigate} title={collapsed ? label : undefined} className={({ isActive }) => `flex min-h-12 items-center gap-3 rounded-xl px-3 text-sm transition-colors ${collapsed ? 'justify-center' : ''} ${isActive ? 'bg-surface text-ink' : 'text-muted hover:bg-surface/60 hover:text-ink'}`}>
                {({ isActive }) => (
                  <>
                    <Icon name={icon} className={`size-5 shrink-0 ${isActive ? 'text-accent' : ''}`} />
                    <span className={collapsed ? 'sr-only' : ''}>{label}</span>
                    {isActive && !collapsed && <span aria-hidden="true" className="ml-auto size-1.5 rounded-full bg-accent" />}
                  </>
                )}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {collapsed ? (
        <button type="button" onClick={onToggle} aria-label="Expand sidebar to chat settings" title="Chat settings" className="mb-6 flex min-h-11 items-center justify-center rounded-lg text-muted hover:bg-surface hover:text-ink">
          <Icon name="settings" />
        </button>
      ) : <SettingsPanel />}

      <div className={`mt-auto shrink-0 border-t border-line/70 pt-5 pb-3 ${collapsed ? 'text-center' : 'px-3'}`}>
        {collapsed ? <span title="ResearchMate" className="text-xs font-semibold text-muted">RM</span> : (
          <>
            <p className="text-xs font-medium text-ink/80">A place for curious minds.</p>
            <p className="mt-1.5 text-xs leading-relaxed text-muted">One question. A new perspective.</p>
          </>
        )}
      </div>
    </div>
  )
}
