import { Icon } from './Icon'
import type { IconName } from './Icon'

type PagePlaceholderProps = { icon: IconName; eyebrow: string; title: string; description: string; note: string }

export function PagePlaceholder({ icon, eyebrow, title, description, note }: PagePlaceholderProps) {
  return (
    <section aria-labelledby="page-heading" className="mx-auto flex w-full max-w-3xl flex-1 flex-col items-center justify-center px-6 py-16 text-center sm:px-10">
      <div className="mb-8 flex size-18 items-center justify-center rounded-3xl border border-accent/15 bg-accent/5 text-accent shadow-[0_0_70px_-20px_#ee9b8830]"><Icon name={icon} className="size-8" /></div>
      <p className="mb-4 text-[10px] font-semibold tracking-[0.22em] text-accent">{eyebrow}</p>
      <h1 id="page-heading" className="text-4xl font-semibold tracking-tight sm:text-5xl">{title}</h1>
      <p className="mt-5 max-w-md text-lg leading-relaxed text-ink/80 sm:text-xl">{description}</p>
      <div aria-hidden="true" className="my-8 h-px w-12 bg-accent/40" />
      <p className="max-w-sm text-sm leading-7 text-muted">{note}</p>
    </section>
  )
}
