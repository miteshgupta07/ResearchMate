import { useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'
import { RoomContext, RoomAudioRenderer, StartAudio, useConnectionState, useIsSpeaking, useLocalParticipant, useRoomContext, useTrackVolume, useVoiceAssistant } from '@livekit/components-react'
import { ConnectionState } from 'livekit-client'
import type { LocalAudioTrack, Room } from 'livekit-client'
import './voiceOrb.css'

type Props = {
  room: Room | null
  syncing: boolean
  error?: string | null
  onEnd: (error?: string | null, source?: Room) => void
  onMicrophoneError: (source: Room) => void
}

const button = 'min-h-11 rounded-full border border-line px-5 text-sm font-medium transition-colors hover:bg-canvas focus-visible:bg-accent/20 disabled:opacity-40'

type OrbState = 'connecting' | 'reconnecting' | 'listening' | 'user-speaking' | 'thinking' | 'assistant-speaking' | 'muted' | 'syncing' | 'error'
const labels: Partial<Record<OrbState, string>> = {
  connecting: 'Connecting…',
  reconnecting: 'Reconnecting…',
  syncing: 'Syncing conversation history…',
  error: 'Voice connection interrupted',
}
const volumeOptions = { fftSize: 64, smoothingTimeConstant: 0.65 }

type AudioTrack = Parameters<typeof useTrackVolume>[0]

// Keep SDK volume updates in this small component instead of rerendering the UI.
function OrbAudioLevel({ track, orbRef }: { track: AudioTrack; orbRef: RefObject<HTMLDivElement | null> }) {
  const level = useTrackVolume(track, volumeOptions)
  const amplitude = Number.isFinite(level) ? Math.max(0, Math.min(1, level * 3)) : 0
  useEffect(() => {
    orbRef.current?.style.setProperty('--voice-level', String(track ? amplitude : 0))
  }, [amplitude, orbRef, track])
  return null
}

function VoiceOrb({ state, audioTrack, error }: { state: OrbState; audioTrack?: AudioTrack; error?: string | null }) {
  const orbRef = useRef<HTMLDivElement>(null)
  const message = error || labels[state]
  return (
    <>
      <div className="voice-orb-stage" aria-hidden="true">
        <div ref={orbRef} className="voice-orb" data-state={state}>
          <OrbAudioLevel track={audioTrack} orbRef={orbRef} />
          <div className="voice-orb__halo" />
          <div className="voice-orb__ripple" />
          <div className="voice-orb__ripple voice-orb__ripple--second" />
          <div className="voice-orb__ripple voice-orb__ripple--third" />
          <div className="voice-orb__surface"><div className="voice-orb__flow" /></div>
        </div>
      </div>
      <h1 className="mt-1 text-2xl font-semibold tracking-tight">Voice Mode</h1>
      {message && <p role={error ? 'alert' : 'status'} aria-atomic="true" className={`mt-3 max-w-sm text-sm leading-6 ${error ? 'text-accent' : 'text-muted'}`}>{message}</p>}
    </>
  )
}

function VoiceActivity({ error, onEnd }: Pick<Props, 'error' | 'onEnd'>) {
  const room = useRoomContext()
  const connection = useConnectionState()
  const { localParticipant, isMicrophoneEnabled, microphoneTrack } = useLocalParticipant()
  const assistant = useVoiceAssistant()
  const userSpeaking = useIsSpeaking(localParticipant)
  const agentReady = ['listening', 'thinking', 'speaking'].includes(assistant.state)

  useEffect(() => {
    if (connection !== ConnectionState.Connected || agentReady) return
    // Legacy useVoiceAssistant has no worker-availability timeout. Reconnection
    // remains owned by the SDK; this only bounds waiting for an unready worker.
    const timeout = window.setTimeout(() => onEnd('The voice assistant is unavailable. Please try again later.', room), 30_000)
    return () => window.clearTimeout(timeout)
  }, [connection, agentReady, onEnd, room])

  let state: OrbState = 'connecting'
  if (error) state = 'error'
  else if (connection === ConnectionState.Reconnecting || connection === ConnectionState.SignalReconnecting) state = 'reconnecting'
  else if (connection === ConnectionState.Connected) {
    if (!isMicrophoneEnabled) state = 'muted'
    // Let the agent state reflect interruptions; microphone noise must not override it.
    else if (assistant.state === 'thinking') state = 'thinking'
    else if (assistant.state === 'speaking') state = 'assistant-speaking'
    else if (userSpeaking) state = 'user-speaking'
    else if (assistant.state === 'listening' || assistant.state === 'idle') state = 'listening'
  }
  const audioTrack = state === 'user-speaking' ? microphoneTrack?.track as LocalAudioTrack | undefined : state === 'assistant-speaking' ? assistant.audioTrack : undefined
  return <VoiceOrb state={state} audioTrack={audioTrack} error={error} />
}

function VoiceControls({ onEnd, onMicrophoneError, error }: Pick<Props, 'onEnd' | 'onMicrophoneError' | 'error'>) {
  const room = useRoomContext()
  const state = useConnectionState()
  const { localParticipant, isMicrophoneEnabled } = useLocalParticipant()
  const [changing, setChanging] = useState(false)
  const changingRef = useRef(false)
  const connected = state === ConnectionState.Connected

  const toggleMicrophone = async () => {
    if (changingRef.current || !connected) return
    changingRef.current = true
    setChanging(true)
    try { await localParticipant.setMicrophoneEnabled(!isMicrophoneEnabled) }
    catch { onMicrophoneError(room) }
    finally { changingRef.current = false; setChanging(false) }
  }

  return (
    <>
      <VoiceActivity error={error} onEnd={onEnd} />
      <div className="voice-controls mt-8 flex flex-wrap justify-center gap-3">
        <button type="button" className={button} disabled={!connected || changing} onClick={() => { void toggleMicrophone() }} aria-label={isMicrophoneEnabled ? 'Mute microphone' : 'Unmute microphone'} aria-pressed={!isMicrophoneEnabled}>
          {isMicrophoneEnabled ? 'Mute' : 'Unmute'}<span className="hidden sm:inline"> microphone</span>
        </button>
        <button type="button" className={`${button} border-accent/30 text-accent`} onClick={() => onEnd(null, room)}>End conversation</button>
      </div>
      <StartAudio label="Enable speaker audio" className={`${button} mt-4`} />
      <RoomAudioRenderer />
    </>
  )
}

export function VoiceMode({ room, syncing, error, onEnd, onMicrophoneError }: Props) {
  const contents = (
    <>
      {room ? <VoiceControls onEnd={onEnd} onMicrophoneError={onMicrophoneError} error={error} /> : <>
        <VoiceOrb state={error ? 'error' : syncing ? 'syncing' : 'connecting'} error={error} />
        {!syncing && <button type="button" className={`${button} mt-8`} onClick={() => onEnd()}>End conversation</button>}
      </>}
    </>
  )
  const className = 'voice-mode-panel mx-auto flex w-full max-w-3xl flex-1 flex-col items-center justify-center px-6 py-6 text-center sm:py-8'
  if (!room) return <section aria-label="Voice Mode" className={className}>{contents}</section>
  return (
    <RoomContext.Provider value={room}>
      <section className={className} aria-label="Voice Mode">{contents}</section>
    </RoomContext.Provider>
  )
}
