import { useSession } from '../../app/session/SessionContext'

export function ModeIndicator() {
  const { mode } = useSession()
  return (
    <div role="status" aria-live="polite" aria-atomic="true" className="px-6 pt-5 sm:px-8">
      <span className="inline-flex items-center gap-2 rounded-full border border-line bg-surface/50 px-3 py-1.5 text-xs text-ink/90">
        <span aria-hidden="true" className="size-1.5 rounded-full bg-accent" />
        <span className="sr-only">Current mode: </span>{mode}
      </span>
    </div>
  )
}
