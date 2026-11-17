import { useRef } from 'react'
import { useSession } from '../../app/session/SessionContext'

export function PdfAttachmentButton({ disabled = false }: { disabled?: boolean }) {
  const pickerRef = useRef<HTMLInputElement>(null)
  const { documentUpload, selectDocument } = useSession()
  const busy = documentUpload.status === 'uploading' || documentUpload.status === 'processing'
  const label = documentUpload.document ? 'Replace PDF attachment' : 'Attach a PDF'
  return (
    <>
      <input ref={pickerRef} type="file" hidden accept=".pdf,application/pdf" disabled={disabled || busy} onChange={(event) => {
        const file = event.target.files?.[0]
        event.target.value = ''
        if (file && !disabled && !busy) selectDocument(file)
      }} />
      <button type="button" aria-label={label} title={label} disabled={disabled || busy} onClick={() => pickerRef.current?.click()} className="flex size-11 shrink-0 items-center justify-center rounded-full text-muted transition-colors hover:bg-canvas hover:text-ink focus-visible:bg-accent/20 disabled:cursor-not-allowed disabled:opacity-40">
        <span aria-hidden="true" className="text-2xl font-light leading-none">+</span>
      </button>
    </>
  )
}

export function PdfAttachmentStatus({ disabled = false }: { disabled?: boolean }) {
  const { documentUpload, removeDocument } = useSession()
  const busy = documentUpload.status === 'uploading' || documentUpload.status === 'processing'
  const filename = documentUpload.document?.filename ?? documentUpload.filename
  if (!filename && !documentUpload.error) return null
  return (
    <div className="px-2 pt-1">
      {filename && (
        <div className="flex max-w-full items-center gap-2 rounded-xl bg-canvas/70 pl-3 pr-1 text-xs">
          <span aria-hidden="true" className="shrink-0 text-accent">PDF</span>
          <span title={filename} className="min-w-0 flex-1 truncate text-ink/90">{filename}</span>
          <span role="status" className="shrink-0 text-muted">{busy ? 'Processing…' : documentUpload.document ? 'Ready' : 'Failed'}</span>
          <button type="button" onClick={removeDocument} disabled={disabled} aria-label={busy ? 'Cancel PDF upload' : 'Remove PDF attachment'} title={busy ? 'Cancel upload' : 'Detach PDF (server file retained)'} className="flex size-11 shrink-0 items-center justify-center rounded-full text-muted hover:text-accent focus-visible:bg-accent/20 disabled:opacity-40"><span aria-hidden="true" className="text-lg">×</span></button>
        </div>
      )}
      {documentUpload.error && <p role="alert" className="px-1 pt-1 text-xs leading-5 text-accent">{documentUpload.error}</p>}
      <span className="sr-only">Removing an attachment detaches it locally. Server files are retained.</span>
    </div>
  )
}
