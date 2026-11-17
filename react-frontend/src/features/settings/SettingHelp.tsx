import { useId, useState } from 'react'
import { Icon } from '../../components/ui/Icon'

export function SettingHelp({ setting, children }: { setting: string; children: string }) {
  const id = useId()
  const [open, setOpen] = useState(false)
  return (
    <span className="relative inline-flex" onMouseEnter={() => setOpen(true)} onMouseLeave={(event) => {
      if (!event.currentTarget.contains(document.activeElement)) setOpen(false)
    }} onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false) }}>
      <button type="button" aria-label={`About ${setting}`} aria-describedby={id} onFocus={() => setOpen(true)} onClick={() => setOpen(true)} onKeyDown={(event) => { if (event.key === 'Escape') setOpen(false) }} className="flex size-7 items-center justify-center rounded-full text-muted hover:bg-surface hover:text-ink focus-visible:text-accent">
        <Icon name="info" className="size-3.5" />
      </button>
      <span id={id} role="tooltip" hidden={!open} className="absolute right-0 top-full z-20 mt-1 w-48 max-w-[calc(100vw-4rem)] rounded-lg border border-line bg-sidebar p-3 text-xs font-normal leading-5 text-ink shadow-xl">{children}</span>
    </span>
  )
}
