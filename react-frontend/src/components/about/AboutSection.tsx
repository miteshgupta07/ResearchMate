import type { ReactNode } from 'react'
import { Icon } from '../ui/Icon'
import type { IconName } from '../ui/Icon'

export function AboutSection({ id, title, icon, children }: { id: string; title: string; icon: IconName; children: ReactNode }) {
  return (
    <section aria-labelledby={id} className="mt-12 sm:mt-16">
      <div className="mb-5 flex items-center gap-3">
        <span className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-accent/10 text-accent"><Icon name={icon} /></span>
        <h2 id={id} className="text-xl font-semibold tracking-tight sm:text-2xl">{title}</h2>
      </div>
      {children}
    </section>
  )
}

export function AboutCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="min-w-0 rounded-2xl border border-line bg-surface/50 p-5 sm:p-6">
      <h3 className="mb-3 text-sm font-semibold text-ink">{title}</h3>
      <div className="text-sm leading-7 text-muted">{children}</div>
    </div>
  )
}
