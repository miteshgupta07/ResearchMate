import { useId, useState } from 'react'

export function ChatInput({ disabled, onSend }: { disabled: boolean; onSend: (text: string) => boolean }) {
  const [draft, setDraft] = useState('')
  const id = useId()
  const send = () => {
    if (!disabled && draft.trim() && onSend(draft)) setDraft('')
  }
  return (
    <form aria-label="Send a chat message" onSubmit={(event) => { event.preventDefault(); send() }} className="mx-auto w-full max-w-3xl px-6 pb-4 pt-3 sm:px-10">
      <label htmlFor={id} className="sr-only">Message</label>
      <div className="flex items-end gap-3 rounded-2xl border border-line bg-surface p-3 focus-within:border-accent/60">
        <textarea id={id} value={draft} onChange={(event) => setDraft(event.target.value)} disabled={disabled} rows={2} placeholder="Ask a question…" aria-describedby={`${id}-help`} onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing && event.nativeEvent.keyCode !== 229) {
            event.preventDefault()
            send()
          }
        }} className="max-h-48 min-h-12 min-w-0 flex-1 resize-y bg-transparent px-1 py-2 text-sm leading-6 text-ink outline-none placeholder:text-muted disabled:opacity-50" />
        <button type="submit" disabled={disabled || !draft.trim()} className="min-h-11 shrink-0 rounded-xl bg-accent px-4 text-sm font-semibold text-sidebar transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40">Send</button>
      </div>
      <p id={`${id}-help`} className="mt-2 text-center text-[11px] text-muted">Enter to send · Shift+Enter for a new line</p>
    </form>
  )
}
