import { Icon } from '../ui/Icon'

type HeaderProps = { title: string; mobileOpen: boolean; onOpenNavigation: () => void }

export function Header({ title, mobileOpen, onOpenNavigation }: HeaderProps) {
  return (
    <header className="flex min-h-20 shrink-0 items-center justify-between gap-3 border-b border-line/60 px-4 group-data-[compact-height=true]/workspace:min-h-14 sm:px-8">
      <div className="flex min-w-0 items-center gap-3">
        <button type="button" onClick={onOpenNavigation} aria-label="Open navigation" aria-expanded={mobileOpen} aria-controls="mobile-sidebar" className="flex size-11 shrink-0 items-center justify-center rounded-lg text-muted hover:bg-surface hover:text-ink md:hidden"><Icon name="panel" /></button>
        <span className="truncate text-sm font-medium">{title}</span>
      </div>
    </header>
  )
}
