import { useRef, useState } from 'react'
import { RoomContext, RoomAudioRenderer, StartAudio, useConnectionState, useLocalParticipant, useRoomContext } from '@livekit/components-react'
import { ConnectionState } from 'livekit-client'
import type { Room } from 'livekit-client'
import { Icon } from '../../components/ui/Icon'

type Props = {
  room: Room | null
  syncing: boolean
  onEnd: (error?: string | null) => void
  onMicrophoneError: () => void
}

const button = 'min-h-11 rounded-full border border-line px-5 text-sm font-medium transition-colors hover:bg-canvas focus-visible:bg-accent/20 disabled:opacity-40'

function VoiceControls({ onEnd, onMicrophoneError }: Pick<Props, 'onEnd' | 'onMicrophoneError'>) {
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
    catch { onMicrophoneError() }
    finally { changingRef.current = false; setChanging(false) }
  }

  return (
    <>
      <p role="status" className="mt-3 text-sm text-muted">{connected ? isMicrophoneEnabled ? 'Connected · Microphone on' : 'Connected · Microphone muted' : state === ConnectionState.Reconnecting ? 'Reconnecting…' : 'Connecting…'}</p>
      <div className="mt-8 flex flex-wrap justify-center gap-3">
        <button type="button" className={button} disabled={!connected || changing} onClick={() => { void toggleMicrophone() }} aria-label={isMicrophoneEnabled ? 'Mute microphone' : 'Unmute microphone'} aria-pressed={!isMicrophoneEnabled}>
          {isMicrophoneEnabled ? 'Mute microphone' : 'Unmute microphone'}
        </button>
        <button type="button" className={`${button} border-accent/30 text-accent`} onClick={() => { void room.disconnect().then(() => onEnd()).catch(() => onEnd()) }}>End conversation</button>
      </div>
      <StartAudio label="Enable speaker audio" className={`${button} mt-4`} />
      <RoomAudioRenderer />
    </>
  )
}

export function VoiceMode({ room, syncing, onEnd, onMicrophoneError }: Props) {
  const contents = (
    <>
      <span className="mx-auto flex size-20 items-center justify-center rounded-full border border-accent/20 bg-accent/10 text-accent"><Icon name="waveform" className="size-9" /></span>
      <h1 className="mt-6 text-2xl font-semibold tracking-tight">Voice Mode</h1>
      {room ? <VoiceControls onEnd={onEnd} onMicrophoneError={onMicrophoneError} /> : <>
        <p role="status" className="mt-3 text-sm text-muted">{syncing ? 'Syncing conversation history…' : 'Connecting…'}</p>
        {!syncing && <button type="button" className={`${button} mt-8`} onClick={() => onEnd()}>End conversation</button>}
      </>}
    </>
  )
  const className = 'mx-auto flex w-full max-w-3xl flex-1 flex-col items-center justify-center px-6 py-10 text-center'
  if (!room) return <section aria-label="Voice Mode" className={className}>{contents}</section>
  return (
    <RoomContext.Provider value={room}>
      <section className={className} aria-label="Voice Mode">{contents}</section>
    </RoomContext.Provider>
  )
}
