import { useId, useLayoutEffect, useRef, useState } from 'react'
import { resizeChatInput, shouldSendOnEnter } from './chatLayout'
import { PdfAttachmentButton, PdfAttachmentStatus } from '../documents/PdfUpload'
import { AgentModeToggle } from '../settings/AgentModeToggle'

export function ChatInput({ disabled, controlsDisabled = false, onSend }: { disabled: boolean; controlsDisabled?: boolean; onSend: (text: string) => boolean }) {
  const [draft, setDraft] = useState('')
  const id = useId()
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  useLayoutEffect(() => {
    if (textareaRef.current) resizeChatInput(textareaRef.current)
  }, [draft])
  useLayoutEffect(() => {
    const textarea = textareaRef.current
    if (!textarea) return
    let previousWidth = textarea.clientWidth
    const resize = () => resizeChatInput(textarea)
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => {
      if (textarea.clientWidth !== previousWidth) {
        previousWidth = textarea.clientWidth
        resize()
      }
    })
    observer?.observe(textarea)
    window.addEventListener('resize', resize)
    window.visualViewport?.addEventListener('resize', resize)
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', resize)
      window.visualViewport?.removeEventListener('resize', resize)
    }
  }, [])
  const send = () => {
    if (!disabled && draft.trim() && onSend(draft)) setDraft('')
  }
  return (
    <form aria-label="Send a chat message" onSubmit={(event) => { event.preventDefault(); send() }} className="mx-auto w-full max-w-3xl shrink-0 px-3 pb-[calc(1rem+env(safe-area-inset-bottom,0px))] pt-2 sm:px-10">
      <label htmlFor={id} className="sr-only">Message</label>
      <div className="chat-composer flex min-w-0 flex-col rounded-3xl border border-line bg-surface p-2 transition-colors focus-within:border-accent/50">
        <PdfAttachmentStatus disabled={controlsDisabled} />
        <textarea ref={textareaRef} id={id} value={draft} onChange={(event) => setDraft(event.target.value)} disabled={disabled} rows={1} placeholder="Ask a question…" aria-describedby={`${id}-help`} onKeyDown={(event) => {
          if (shouldSendOnEnter({ key: event.key, shiftKey: event.shiftKey, isComposing: event.nativeEvent.isComposing, keyCode: event.nativeEvent.keyCode })) {
            event.preventDefault()
            send()
          }
        }} className="chat-scrollbar min-h-10 max-h-[min(12rem,30dvh,calc(var(--workspace-height,100dvh)*0.25))] w-full min-w-0 shrink-0 resize-none border-0 bg-transparent px-3 py-2 text-base leading-6 text-ink outline-none placeholder:text-muted focus-visible:outline-none disabled:opacity-50 sm:text-sm" />
        <div className="flex shrink-0 items-center justify-between gap-2">
          <div className="flex min-w-0 items-center gap-1">
            <PdfAttachmentButton disabled={controlsDisabled} />
            <AgentModeToggle />
          </div>
          <button type="submit" disabled={disabled || !draft.trim()} aria-label="Send message" className="flex size-11 shrink-0 items-center justify-center rounded-full bg-accent text-sidebar transition-colors hover:bg-ink focus-visible:bg-ink disabled:cursor-not-allowed disabled:opacity-40"><span aria-hidden="true" className="text-xl">↑</span></button>
        </div>
      </div>
      <p id={`${id}-help`} className="sr-only">Enter to send. Shift+Enter for a new line.</p>
    </form>
  )
}
