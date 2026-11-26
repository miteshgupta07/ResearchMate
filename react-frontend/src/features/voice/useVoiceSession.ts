import { useCallback, useEffect, useRef, useState } from 'react'
import { useSession } from '../../app/session/SessionContext'
import { createVoiceSession } from '../../services/api/voice'
import { ApiError } from '../../services/api/errors'
import type { Room } from 'livekit-client'

type Phase = 'idle' | 'connecting' | 'connected' | 'syncing'
type Attempt = { controller: AbortController; joined: boolean; room?: Room }

function wait(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const abort = () => { clearTimeout(timer); reject(new DOMException('Cancelled', 'AbortError')) }
    const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve() }, ms)
    if (signal.aborted) abort()
    else signal.addEventListener('abort', abort, { once: true })
  })
}

/** The backend has no flush acknowledgement. Re-read snapshots for a bounded grace period. */
export async function refreshVoiceHistory(refresh: () => Promise<void>, signal: AbortSignal) {
  for (const delay of [250, 750, 1500, 2500, 4000]) {
    await wait(delay, signal)
    if (signal.aborted) return
    await refresh()
  }
}

function connectionError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 429) return 'Too many voice connection attempts. Wait a minute and try again.'
    if (error.status === 403) return 'Voice Mode is unavailable from this site.'
    if (error.status === 503) return 'Voice Mode is temporarily unavailable. Please try again later.'
  }
  return 'Could not connect to Voice Mode. Check your connection and try again.'
}

export function useVoiceSession(sessionId: string, refresh: () => Promise<void>) {
  const { voiceHistoryPending, setVoiceHistoryPending } = useSession()
  const [phase, setPhase] = useState<Phase>('idle')
  const [room, setRoom] = useState<Room | null>(null)
  const [error, setError] = useState<string | null>(null)
  const attempt = useRef<Attempt | null>(null)
  const syncing = useRef<AbortController | null>(null)
  const generation = useRef(0)

  const reconcile = useCallback(async () => {
    if (syncing.current) return
    const controller = new AbortController()
    syncing.current = controller
    try {
      await Promise.resolve()
      if (controller.signal.aborted) return
      setPhase('syncing')
      await refreshVoiceHistory(refresh, controller.signal)
      if (!controller.signal.aborted) {
        setVoiceHistoryPending(false)
        setPhase('idle')
      }
    } catch {
      if (!controller.signal.aborted) {
        setError('Conversation history could not be refreshed. Check history before continuing.')
        setVoiceHistoryPending(false)
        setPhase('idle')
      }
    } finally {
      if (syncing.current === controller) syncing.current = null
    }
  }, [refresh, setVoiceHistoryPending])

  const finish = useCallback((message: string | null = null) => {
    const current = attempt.current
    if (!current) return
    attempt.current = null
    current.controller.abort()
    setError(message)
    const token = generation.current
    void (async () => {
      await current.room?.disconnect().catch(() => {})
      if (generation.current !== token) return
      setRoom(null)
      if (current.joined) void reconcile()
      else setPhase('idle')
    })()
  }, [reconcile])

  const start = useCallback(async () => {
    if (attempt.current || syncing.current || voiceHistoryPending) return
    const current: Attempt = { controller: new AbortController(), joined: false }
    attempt.current = current
    setError(null)
    setPhase('connecting')
    try {
      const { Room, RoomEvent, DisconnectReason } = await import('livekit-client')
      if (attempt.current !== current || current.controller.signal.aborted) return
      current.room = new Room()
      current.room.on(RoomEvent.Disconnected, (reason) => finish(reason === DisconnectReason.CLIENT_INITIATED ? null : 'The voice connection was lost. Reconnect to continue.'))
      const details = await createVoiceSession({ session_id: sessionId }, { signal: current.controller.signal })
      if (attempt.current !== current || current.controller.signal.aborted) return
      current.joined = true
      setVoiceHistoryPending(true)
      setRoom(current.room)
      await current.room.connect(details.url, details.token)
      if (current.controller.signal.aborted) return
      try { await current.room.localParticipant.setMicrophoneEnabled(true) }
      catch { if (attempt.current === current) finish('Microphone access failed. Allow microphone access in your browser and try again.'); return }
      if (!current.controller.signal.aborted) setPhase('connected')
    } catch (cause) {
      if (attempt.current === current) finish(connectionError(cause))
    } finally {
      // The SDK also stops a track if publication fails after delayed device access.
      if (current.controller.signal.aborted) void current.room?.disconnect().catch(() => {})
    }
  }, [sessionId, voiceHistoryPending, setVoiceHistoryPending, finish])

  const microphoneFailed = useCallback(() => finish('Microphone access failed. Allow microphone access in your browser and try again.'), [finish])

  useEffect(() => {
    if (voiceHistoryPending && !attempt.current) void reconcile()
  }, [voiceHistoryPending, reconcile])

  useEffect(() => () => {
    generation.current += 1
    const current = attempt.current
    attempt.current = null
    current?.controller.abort()
    void current?.room?.disconnect().catch(() => {})
    syncing.current?.abort()
    syncing.current = null
    // The SDK disconnect stops published tracks; refresh pending history on return.
  }, [sessionId])

  return { phase, room, error, start, finish, microphoneFailed, busy: phase !== 'idle' || voiceHistoryPending }
}
