import { uploadDocument } from '../../services/api/documents.ts'
import { ApiError } from '../../services/api/errors.ts'
import type { RequestOptions } from '../../services/api/client.ts'
import type { DocumentUploadResponse } from '../../types/api.ts'

export interface AttachedDocument {
  document_id: string
  filename: string
}

export interface DocumentUploadState {
  status: 'idle' | 'uploading' | 'processing' | 'success' | 'error'
  document: AttachedDocument | null
  filename: string | null
  error: string | null
}

type Uploader = (file: File, options: RequestOptions) => Promise<DocumentUploadResponse>

export function validatePdf(file: File): string | null {
  if (!file.name.toLowerCase().endsWith('.pdf') || (file.type && file.type !== 'application/pdf')) {
    return 'Choose a PDF file (.pdf, application/pdf).'
  }
  if (file.size === 0) return 'This PDF is empty. Choose a file with content.'
  return null
}

function uploadError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.kind === 'timeout') return 'The upload timed out. Select the PDF again to retry.'
    if (error.kind === 'network') return 'Unable to upload the PDF. Check your connection and select it again.'
    if (error.kind === 'configuration') return 'Uploads are unavailable. Check VITE_API_BASE_URL.'
    if (error.status === 400 || error.status === 422) return 'The server could not accept this PDF. Check the file and select it again.'
  }
  return 'The PDF could not be processed. Select it again to retry.'
}

export function createDocumentStore(uploader: Uploader = uploadDocument) {
  let state: DocumentUploadState = { status: 'idle', document: null, filename: null, error: null }
  let generation = 0
  let controller: AbortController | undefined
  let selectedFile: File | undefined
  const listeners = new Set<() => void>()
  const publish = (next: DocumentUploadState) => {
    state = next
    listeners.forEach((listener) => listener())
  }
  const invalidate = () => {
    generation += 1
    controller?.abort()
    controller = undefined
    selectedFile = undefined
  }
  const remove = () => {
    invalidate()
    publish({ status: 'idle', document: null, filename: null, error: null })
  }

  const select = (file: File): boolean => {
    if (controller || (selectedFile === file && state.document)) return false
    const error = validatePdf(file)
    if (error) {
      // An invalid picker selection does not discard a previously processed PDF.
      publish({ ...state, status: 'error', error })
      return false
    }
    invalidate()
    const token = generation
    controller = new AbortController()
    const signal = controller.signal
    selectedFile = file
    // A valid replacement clears the previous ID immediately, including on failure.
    publish({ status: 'uploading', document: null, filename: file.name, error: null })
    void (async () => {
      try {
        // Browsers with an unknown PDF MIME otherwise send application/octet-stream.
        const pdf = file.type ? file : new File([file], file.name, { type: 'application/pdf', lastModified: file.lastModified })
        const request = uploader(pdf, { signal })
        if (generation !== token || signal.aborted) return
        // Fetch has no upload-progress event: this phase includes transfer and ingestion.
        publish({ status: 'processing', document: null, filename: file.name, error: null })
        const response = await request
        if (generation !== token || signal.aborted) return
        if (!response || typeof response.document_id !== 'string' || !response.document_id.trim()
          || typeof response.filename !== 'string' || !response.filename.trim() || response.status !== 'processed') {
          throw new Error('Invalid document upload response')
        }
        publish({ status: 'success', document: { document_id: response.document_id, filename: response.filename }, filename: response.filename, error: null })
      } catch (error) {
        if (generation !== token || signal.aborted) return
        selectedFile = undefined
        publish({ status: 'error', document: null, filename: file.name, error: uploadError(error) })
      } finally {
        if (generation === token) controller = undefined
      }
    })()
    return true
  }

  return {
    getSnapshot: () => state,
    subscribe: (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener) } },
    select, remove,
    // Preserve the existing context setter for future integrations; no server action.
    setDocumentId: (id: string | null) => {
      if (!id) { remove(); return }
      const filename = state.document?.filename ?? 'Attached PDF'
      invalidate()
      publish({ status: 'success', document: { document_id: id, filename }, filename, error: null })
    },
  }
}
