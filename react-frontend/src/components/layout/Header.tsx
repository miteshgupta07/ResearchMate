import { Icon } from '../ui/Icon'

type HeaderProps = { title: string; mobileOpen: boolean; onOpenNavigation: () => void }

export function Header({ title, mobileOpen, onOpenNavigation }: HeaderProps) {
  return (
    <header className="flex min-h-20 shrink-0 items-center justify-between gap-3 border-b border-line/60 px-4 sm:px-8">
      <div className="flex min-w-0 items-center gap-3">
        <button type="button" onClick={onOpenNavigation} aria-label="Open navigation" aria-expanded={mobileOpen} aria-controls="mobile-sidebar" className="flex size-11 shrink-0 items-center justify-center rounded-lg text-muted hover:bg-surface hover:text-ink md:hidden"><Icon name="panel" /></button>
        <div className="flex min-w-0 items-center gap-3 text-sm">
          <span className="hidden text-muted sm:inline">Workspace</span>
          <span aria-hidden="true" className="hidden text-line sm:inline">/</span>
          <span className="truncate font-medium">{title}</span>
        </div>
      </div>
      <span className="shrink-0 rounded-full border border-line bg-surface/40 px-3 py-1 text-[11px] font-medium tracking-wide text-muted">Preview</span>
    </header>
  )
}
