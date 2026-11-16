import { useId } from 'react'
import { useSession } from '../../app/session/SessionContext'

export function PdfUpload({ disabled = false }: { disabled?: boolean }) {
  const id = useId()
  const { documentUpload, selectDocument, removeDocument } = useSession()
  const busy = documentUpload.status === 'uploading' || documentUpload.status === 'processing'
  const filename = documentUpload.document?.filename ?? documentUpload.filename

  return (
    <section aria-label="PDF attachment" className="mx-auto w-full max-w-3xl px-6 pt-3 sm:px-10">
      <div className="rounded-xl border border-line bg-surface/40 px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="min-w-0 flex-1 break-words text-sm text-ink [overflow-wrap:anywhere]">{filename ?? 'Attach a PDF'}</p>
          <div className="flex items-center gap-3">
            <label htmlFor={id} className="sr-only">{documentUpload.document ? 'Replace PDF' : 'Choose PDF'}</label>
            <input id={id} type="file" accept=".pdf,application/pdf" disabled={disabled || busy} aria-describedby={`${id}-status ${id}-help`} onChange={(event) => {
              const file = event.target.files?.[0]
              // Clear the picker so a failed or same-name file can be selected again.
              event.target.value = ''
              if (file && !disabled) selectDocument(file)
            }} className="min-h-11 w-full max-w-56 text-xs text-muted file:mr-3 file:min-h-11 file:cursor-pointer file:rounded-lg file:border file:border-line file:bg-surface file:px-3 file:text-xs file:text-ink disabled:opacity-50" />
            {(filename || busy) && <button type="button" onClick={removeDocument} disabled={disabled} className="min-h-11 rounded-lg px-2 text-xs text-accent hover:bg-accent/10 disabled:opacity-50">{busy ? 'Cancel upload' : 'Remove'}</button>}
          </div>
        </div>
        <p id={`${id}-status`} role={documentUpload.error ? 'alert' : 'status'} aria-live={documentUpload.error ? 'assertive' : 'polite'} className={`mt-1 text-xs leading-5 ${documentUpload.error ? 'text-accent' : 'text-muted'}`}>
          {documentUpload.error ?? (documentUpload.status === 'uploading' ? 'Sending PDF…'
            : documentUpload.status === 'processing' ? 'Uploading and processing PDF…'
              : documentUpload.document ? 'PDF processed and attached. Document Q&A is coming in a future update.' : 'Choose a non-empty PDF to prepare a document.')}
        </p>
        <p id={`${id}-help`} className="mt-1 text-[11px] leading-5 text-muted">Removing a PDF detaches it from this session. Server files are retained.</p>
      </div>
    </section>
  )
}
