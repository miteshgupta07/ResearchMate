import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'

type Message = { role: string; content: string }

async function setup(page: Page, options: { historyDelay?: number; tokenDelay?: number; tokenStatus?: number; textDelay?: number; textStatus?: number; uploadDelay?: number } = {}) {
  const calls: { history: number; historyWaits: number; voice: string[]; chat: string[] } = { history: 0, historyWaits: 0, voice: [], chat: [] }
  const messages: Message[] = []
  await page.route('http://127.0.0.1:8765/**', async (route) => {
    const request = route.request()
    const headers = { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Methods': 'GET, POST, DELETE' }
    if (request.method() === 'OPTIONS') return route.fulfill({ status: 204, headers })
    const url = new URL(request.url())
    let body: unknown
    let status = 200
    if (url.pathname === '/history') {
      calls.history += 1
      if (url.searchParams.get('wait_for_idle') === 'true') calls.historyWaits += 1
      if (options.historyDelay) await new Promise((resolve) => setTimeout(resolve, options.historyDelay))
      body = { session_id: url.searchParams.get('session_id'), messages }
    } else if (url.pathname === '/voice/session') {
      calls.voice.push(request.postDataJSON().session_id)
      if (options.tokenDelay) await new Promise((resolve) => setTimeout(resolve, options.tokenDelay))
      status = options.tokenStatus ?? 200
      body = status === 200 ? { url: 'wss://voice.test', room_name: 'test-room', token: 'test-voice-token' } : { detail: 'Unavailable' }
    } else if (url.pathname === '/documents/upload') {
      if (options.uploadDelay) await new Promise((resolve) => setTimeout(resolve, options.uploadDelay))
      body = { document_id: 'pdf-1', filename: 'paper.pdf', status: 'processed' }
    } else {
      if (options.textDelay) await new Promise((resolve) => setTimeout(resolve, options.textDelay))
      const data = request.postDataJSON()
      calls.chat.push(url.pathname)
      status = options.textStatus ?? 200
      if (status === 200) messages.push({ role: 'user', content: data.message }, { role: 'assistant', content: 'Text answer' })
      body = status === 200 ? { role: 'assistant', content: 'Text answer', sources: [], intent: 'INTENT_CHAT', confidence: 1, metadata: {} } : { detail: 'Conversation is busy.' }
    }
    await route.fulfill({ status, headers, contentType: 'application/json', body: JSON.stringify(body) })
  })
  await page.goto('/')
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeVisible()
  return { calls, messages }
}

/** Keep real LiveKit React hooks/audio renderer; stub signaling and publishing only. */
async function mockRoom(page: Page, options: { deny?: boolean; fail?: boolean; microphoneDelay?: number; connectDelay?: number; disconnectDelay?: number } = {}) {
  await page.evaluate(async (flags) => {
    const source = await (await fetch('/src/features/voice/useVoiceSession.ts')).text()
    const path = source.match(/import\("([^"]*livekit-client[^"]*)"\)/)?.[1]
    if (!path) throw new Error('SDK module URL was not found')
    const sdk = await import(path) as typeof import('livekit-client')
    const state = { connections: 0, disconnects: 0, mic: false, stream: null as MediaStream | null, room: null as import('livekit-client').Room | null }
    Object.assign(window, { voiceTest: state })
    const connect = async function (this: import('livekit-client').Room) {
      state.connections += 1
      state.room = this
      if (flags.fail) throw new Error('Simulated signaling failure')
      Object.defineProperty(this.localParticipant, 'isMicrophoneEnabled', { configurable: true, get: () => state.mic })
      this.localParticipant.setMicrophoneEnabled = async (enabled: boolean) => {
        if (flags.microphoneDelay) await new Promise((resolve) => setTimeout(resolve, flags.microphoneDelay))
        if (flags.deny) throw new DOMException('Denied', 'NotAllowedError')
        if (enabled && (!state.stream || state.stream.getTracks().every((track) => track.readyState === 'ended'))) {
          state.stream = await navigator.mediaDevices.getUserMedia({ audio: true })
          const track = new sdk.LocalAudioTrack(state.stream.getAudioTracks()[0])
          track.mediaStream = state.stream
          track.source = sdk.Track.Source.Microphone
          const publication = new sdk.LocalTrackPublication(sdk.Track.Kind.Audio, { sid: 'TR_capture', source: 2, muted: false, name: 'microphone' } as never, track)
          this.localParticipant.trackPublications.set(publication.trackSid, publication)
          this.localParticipant.audioTrackPublications.set(publication.trackSid, publication)
        }
        if (this.state === sdk.ConnectionState.Disconnected) {
          state.stream?.getTracks().forEach((track) => track.stop())
          throw new Error('Disconnected during device access')
        }
        state.mic = enabled
        this.localParticipant.emit(sdk.ParticipantEvent.ParticipantPermissionsChanged)
        return undefined
      }
      this.state = sdk.ConnectionState.Connecting
      if (flags.connectDelay) await new Promise((resolve) => setTimeout(resolve, flags.connectDelay))
      if (this.state === sdk.ConnectionState.Disconnected) throw new Error('Cancelled during connect')
      this.state = sdk.ConnectionState.Connected
      this.emit(sdk.RoomEvent.ConnectionStateChanged, sdk.ConnectionState.Connected)
      this.emit(sdk.RoomEvent.Connected)
    }
    const disconnect = async function (this: import('livekit-client').Room) {
      if (this.state === sdk.ConnectionState.Disconnected) return
      state.disconnects += 1
      if (flags.disconnectDelay) await new Promise((resolve) => setTimeout(resolve, flags.disconnectDelay))
      state.mic = false
      state.stream?.getTracks().forEach((track) => track.stop())
      this.state = sdk.ConnectionState.Disconnected
      this.emit(sdk.RoomEvent.ConnectionStateChanged, sdk.ConnectionState.Disconnected)
      this.emit(sdk.RoomEvent.Disconnected, sdk.DisconnectReason.CLIENT_INITIATED)
    }
    // Room defines these callbacks as constructor-assigned fields.
    Object.defineProperty(sdk.Room.prototype, 'connect', { configurable: true, get: () => connect, set: () => {} })
    Object.defineProperty(sdk.Room.prototype, 'disconnect', { configurable: true, get: () => disconnect, set: () => {} })
  }, options)
}

/** Real SDK participants, publications and hooks, with generated audio for the visual meter. */
async function addVisualAgent(page: Page) {
  await page.evaluate(async () => {
    const source = await (await fetch('/src/features/voice/useVoiceSession.ts')).text()
    const path = source.match(/import\("([^"]*livekit-client[^"]*)"\)/)?.[1]
    if (!path) throw new Error('SDK module URL was not found')
    const sdk = await import(path) as typeof import('livekit-client')
    const room = Reflect.get(window, 'voiceTest').room as import('livekit-client').Room
    let attributes = { 'lk.agent.state': 'listening' }
    const agent = new sdk.RemoteParticipant({} as never, 'PA_visual', 'researchmate-voice', undefined, undefined, attributes, undefined, sdk.ParticipantKind.AGENT)
    Object.defineProperty(agent, 'attributes', { get: () => attributes })
    const context = new AudioContext()
    await context.resume()
    const tone = () => {
      const oscillator = context.createOscillator()
      oscillator.frequency.value = 440
      const gain = context.createGain()
      gain.gain.value = 0
      const output = context.createMediaStreamDestination()
      oscillator.connect(gain).connect(output)
      oscillator.start()
      return { oscillator, gain, stream: output.stream }
    }
    const local = tone()
    const remote = tone()
    const localTrack = new sdk.LocalAudioTrack(local.stream.getAudioTracks()[0])
    localTrack.mediaStream = local.stream
    localTrack.source = sdk.Track.Source.Microphone
    const localPublication = new sdk.LocalTrackPublication(sdk.Track.Kind.Audio, { sid: 'TR_local', source: 2, muted: false, name: 'microphone' } as never, localTrack)
    localPublication.source = sdk.Track.Source.Microphone
    room.localParticipant.audioTrackPublications.set(localPublication.trackSid, localPublication)
    room.localParticipant.trackPublications.set(localPublication.trackSid, localPublication)
    room.localParticipant.emit(sdk.ParticipantEvent.ParticipantPermissionsChanged)
    const remoteTrack = new sdk.RemoteAudioTrack(remote.stream.getAudioTracks()[0], 'TR_remote', undefined as never)
    remoteTrack.mediaStream = remote.stream
    remoteTrack.source = sdk.Track.Source.Microphone
    const remotePublication = new sdk.RemoteTrackPublication(sdk.Track.Kind.Audio, { sid: 'TR_remote', source: 2, muted: false, name: 'agent-audio', mimeType: 'audio/opus' } as never, true)
    remotePublication.setTrack(remoteTrack)
    agent.audioTrackPublications.set(remotePublication.trackSid, remotePublication)
    agent.trackPublications.set(remotePublication.trackSid, remotePublication)
    room.remoteParticipants.set(agent.identity, agent)
    room.emit(sdk.RoomEvent.ParticipantConnected, agent)
    room.emit(sdk.RoomEvent.TrackSubscribed, remoteTrack, remotePublication, agent)
    Object.assign(Reflect.get(window, 'voiceTest'), {
      setAgentState: (state: string) => {
        attributes = { 'lk.agent.state': state }
        agent.emit(sdk.ParticipantEvent.AttributesChanged, attributes)
      },
      setUserSpeaking: (speaking: boolean) => room.localParticipant.setIsSpeaking(speaking),
      setConnectionState: (state: import('livekit-client').ConnectionState) => {
        room.state = state
        room.emit(sdk.RoomEvent.ConnectionStateChanged, state)
      },
      setLevel: (speaker: string, value: number) => (speaker === 'local' ? local : remote).gain.gain.setValueAtTime(value, context.currentTime),
    })
    room.once(sdk.RoomEvent.Disconnected, () => {
      local.oscillator.stop(); remote.oscillator.stop()
      localTrack.stop(); remoteTrack.stop()
      void context.close()
    })
  })
}

test('orb follows SDK states, audio amplitude, interruption and mute', async ({ page }) => {
  await setup(page)
  await mockRoom(page)
  await page.getByRole('button', { name: 'Start Voice Mode' }).click()
  await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeEnabled()
  const orb = page.locator('.voice-orb')
  await expect(orb).toHaveAttribute('data-state', 'connecting')
  await expect(page.getByRole('region', { name: 'Voice Mode' }).getByRole('status')).toContainText('Connecting')
  await addVisualAgent(page)
  await expect(orb).toHaveAttribute('data-state', 'listening')
  await expect(page.getByRole('region', { name: 'Voice Mode' }).getByRole('status')).toHaveCount(0)
  await expect(page.locator('.voice-orb__surface')).toHaveCSS('animation-name', 'voice-orb-breathe')
  await expect(page.locator('.voice-orb__flow')).toHaveCSS('animation-name', 'voice-orb-flow')
  const flowTransform = await page.locator('.voice-orb__flow').evaluate((node) => getComputedStyle(node).transform)
  await expect.poll(() => page.locator('.voice-orb__flow').evaluate((node) => getComputedStyle(node).transform)).not.toBe(flowTransform)
  await expect(page.locator('.opening-splash')).toHaveCount(0)
  await page.screenshot({ path: 'node_modules/.cache/voice-orb-desktop.png' })
  await page.evaluate(() => Reflect.get(window, 'voiceTest').setAgentState('thinking'))
  await expect(orb).toHaveAttribute('data-state', 'thinking')
  await expect(page.getByRole('region', { name: 'Voice Mode' }).getByRole('status')).toHaveCount(0)
  await expect(page.locator('.voice-orb__surface')).toHaveCSS('animation-name', 'voice-orb-think')
  await page.evaluate(() => {
    Reflect.get(window, 'voiceTest').setAgentState('speaking')
    Reflect.get(window, 'voiceTest').setLevel('remote', 0.03)
  })
  await expect(orb).toHaveAttribute('data-state', 'assistant-speaking')
  await expect.poll(() => orb.evaluate((node) => Number((node as HTMLElement).style.getPropertyValue('--voice-level')))).toBeGreaterThan(0.05)
  await expect(page.locator('.voice-orb__ripple').first()).toHaveCSS('animation-name', 'voice-orb-ripple')
  await page.evaluate(() => {
    Reflect.get(window, 'voiceTest').setLevel('local', 0.03)
    Reflect.get(window, 'voiceTest').setUserSpeaking(true)
  })
  // Microphone noise/activity cannot replace the assistant's speaking animation.
  await expect(orb).toHaveAttribute('data-state', 'assistant-speaking')
  await page.evaluate(() => Reflect.get(window, 'voiceTest').setAgentState('listening'))
  await expect(orb).toHaveAttribute('data-state', 'user-speaking')
  await expect(page.getByRole('region', { name: 'Voice Mode' }).getByRole('status')).toHaveCount(0)
  await expect.poll(() => orb.evaluate((node) => Number((node as HTMLElement).style.getPropertyValue('--voice-level')))).toBeGreaterThan(0.05)
  await page.evaluate(() => {
    Reflect.get(window, 'voiceTest').setLevel('local', 0)
    Reflect.get(window, 'voiceTest').setUserSpeaking(false)
    Reflect.get(window, 'voiceTest').setAgentState('speaking')
  })
  await expect(orb).toHaveAttribute('data-state', 'assistant-speaking')
  await page.getByRole('button', { name: 'Mute microphone', exact: true }).click()
  await expect(orb).toHaveAttribute('data-state', 'muted')
  await expect(page.getByRole('button', { name: 'Unmute microphone', exact: true })).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByRole('region', { name: 'Voice Mode' }).getByRole('status')).toHaveCount(0)
  await expect(page.locator('.voice-orb__surface')).toHaveCSS('animation-name', 'none')
  await page.getByRole('button', { name: 'Unmute microphone', exact: true }).click()
  await expect(orb).toHaveAttribute('data-state', 'assistant-speaking')
  await page.evaluate(() => Reflect.get(window, 'voiceTest').setConnectionState('reconnecting'))
  await expect(orb).toHaveAttribute('data-state', 'reconnecting')
  await expect(page.getByRole('region', { name: 'Voice Mode' }).getByRole('status')).toContainText('Reconnecting')
  await page.evaluate(() => Reflect.get(window, 'voiceTest').setConnectionState('connected'))
  await expect(orb).toHaveAttribute('data-state', 'assistant-speaking')
  await page.getByRole('button', { name: 'End conversation', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeEnabled({ timeout: 15_000 })
})

test('orb honors reduced motion without hiding state updates', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await setup(page)
  await mockRoom(page)
  await page.getByRole('button', { name: 'Start Voice Mode' }).click()
  await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeEnabled()
  await addVisualAgent(page)
  await expect(page.locator('.voice-orb')).toHaveAttribute('data-state', 'listening')
  await page.evaluate(() => Reflect.get(window, 'voiceTest').setAgentState('thinking'))
  await expect(page.locator('.voice-orb')).toHaveAttribute('data-state', 'thinking')
  for (const layer of ['.voice-orb__surface', '.voice-orb__halo', '.voice-orb__flow']) {
    await expect(page.locator(layer)).toHaveCSS('animation-name', 'none')
    await expect(page.locator(layer)).toHaveCSS('transform', 'none')
  }
  await expect(page.locator('.voice-orb__ripple').first()).toHaveCSS('display', 'none')
  await page.getByRole('button', { name: 'Mute microphone', exact: true }).click()
  await expect(page.locator('.voice-orb')).toHaveAttribute('data-state', 'muted')
  await expect(page.getByRole('button', { name: 'Unmute microphone', exact: true })).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByRole('region', { name: 'Voice Mode' }).getByRole('status')).toHaveCount(0)
})

test('orb and controls fit a compact mobile chat workspace', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 })
  await setup(page)
  await mockRoom(page)
  await page.getByRole('button', { name: 'Start Voice Mode' }).click()
  await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeEnabled()
  const orb = await page.locator('.voice-orb').boundingBox()
  const workspace = await page.getByRole('region', { name: 'Chat messages' }).boundingBox()
  expect(orb && workspace).toBeTruthy()
  expect(Math.abs(orb!.x + orb!.width / 2 - (workspace!.x + workspace!.width / 2))).toBeLessThan(2)
  await expect(page.getByRole('button', { name: 'End conversation', exact: true })).toBeInViewport()
  expect(await page.getByRole('region', { name: 'Chat messages' }).evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true)
  await expect(page.locator('.opening-splash')).toHaveCount(0)
  await page.screenshot({ path: 'node_modules/.cache/voice-orb-mobile.png' })
})

test('composer switches between voice, text, Agent and PDF modes', async ({ page }) => {
  const { calls } = await setup(page)
  const voice = page.getByRole('button', { name: 'Start Voice Mode' })
  await expect(voice).toBeEnabled()
  const bounds = await voice.boundingBox()
  await page.getByRole('textbox', { name: 'Message', exact: true }).fill('A text question')
  const send = page.getByRole('button', { name: 'Send message' })
  expect(await send.boundingBox()).toEqual(bounds)
  await send.click()
  await expect(page.getByText('Text answer', { exact: true })).toHaveCount(1)
  await expect(voice).toBeEnabled()
  await page.getByRole('switch', { name: 'Agent Mode' }).click()
  await expect(send).toBeVisible()
  await expect(send).toBeDisabled()
  await page.getByRole('switch', { name: 'Agent Mode' }).click()
  await page.locator('input[type=file]').setInputFiles({ name: 'paper.pdf', mimeType: 'application/pdf', buffer: Buffer.from('%PDF-1.4\nTest') })
  await expect(page.getByText('Ready', { exact: true })).toBeVisible()
  await page.getByRole('textbox', { name: 'Message', exact: true }).fill('PDF question')
  await send.click()
  await expect(page.getByText('Text answer', { exact: true })).toHaveCount(2)
  await page.getByRole('switch', { name: 'Agent Mode' }).click()
  await page.getByRole('textbox', { name: 'Message', exact: true }).fill('Agent question')
  await send.click()
  await expect(page.getByText('Text answer', { exact: true })).toHaveCount(3)
  await expect(send).toBeVisible()
  await expect(voice).toHaveCount(0)
  await page.getByRole('button', { name: 'Remove PDF attachment' }).click()
  await page.getByRole('switch', { name: 'Agent Mode' }).click()
  await expect(voice).toBeEnabled()
  expect(calls.chat).toEqual(['/chat', '/rag/query', '/agent/route'])
})

test('voice is disabled during initial history and text requests', async ({ page }) => {
  await setup(page, { historyDelay: 1500, textDelay: 900 })
  const voice = page.getByRole('button', { name: 'Start Voice Mode' })
  await expect(voice).toBeDisabled()
  await expect(voice).toBeEnabled()
  await page.getByRole('textbox', { name: 'Message', exact: true }).fill('Pending text request')
  await page.getByRole('button', { name: 'Send message' }).click()
  await expect(voice).toBeDisabled()
  await expect(voice).toBeEnabled()
})

test('text conflicts show a useful error without duplicating an unsaved message', async ({ page }) => {
  await setup(page, { textStatus: 409 })
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeEnabled()
  await page.getByRole('textbox', { name: 'Message', exact: true }).fill('Concurrent text question')
  await page.getByRole('button', { name: 'Send message' }).click()
  await expect(page.getByRole('alert')).toContainText('conversation is busy')
  await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toBeEnabled()
  await expect(page.getByText('Concurrent text question', { exact: true })).toHaveCount(0)
})

test('voice is disabled during a PDF upload and stays unavailable after attachment', async ({ page }) => {
  await setup(page, { uploadDelay: 1500 })
  await page.locator('input[type=file]').setInputFiles({ name: 'paper.pdf', mimeType: 'application/pdf', buffer: Buffer.from('%PDF-1.4\nTest') })
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeDisabled()
  await expect(page.getByText('Ready', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Send message' })).toBeDisabled()
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toHaveCount(0)
})

test('connecting locks controls, cancel prevents a late connection', async ({ page }) => {
  const { calls } = await setup(page, { tokenDelay: 800 })
  await mockRoom(page)
  await page.getByRole('button', { name: 'Start Voice Mode' }).dblclick()
  await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toBeDisabled()
  await expect(page.getByRole('switch', { name: 'Agent Mode' })).toBeDisabled()
  await expect(page.getByRole('button', { name: 'Attach a PDF' })).toBeDisabled()
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeDisabled()
  await expect.poll(() => calls.voice.length).toBe(1)
  await page.getByRole('button', { name: 'End conversation', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeEnabled()
  await page.waitForTimeout(900)
  expect(await page.evaluate(() => Reflect.get(window, 'voiceTest').connections)).toBe(0)
})

test('microphone toggle, ordered history refresh and resource cleanup', async ({ page }) => {
  const { calls, messages } = await setup(page, { historyDelay: 1500 })
  await mockRoom(page)
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeEnabled()
  await page.getByRole('button', { name: 'Start Voice Mode' }).click()
  await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeEnabled()
  await page.getByRole('button', { name: 'Mute microphone', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Unmute microphone', exact: true })).toBeEnabled()
  await page.getByRole('button', { name: 'Unmute microphone', exact: true }).click()
  const historyBefore = calls.history
  await page.getByRole('button', { name: 'End conversation', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeDisabled()
  await expect.poll(() => calls.historyWaits).toBe(1)
  messages.push({ role: 'user', content: 'Voice question' }, { role: 'assistant', content: 'Voice answer' })
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeEnabled({ timeout: 15_000 })
  await expect(page.getByText('Voice answer', { exact: true })).toHaveCount(1)
  await expect(page.getByText('Voice question', { exact: true })).toHaveCount(1)
  expect(calls.history - historyBefore).toBe(1)
  expect(calls.historyWaits).toBe(1)
  expect(calls.voice).toHaveLength(1)
  expect(await page.evaluate(() => Reflect.get(window, 'voiceTest').stream.getTracks().every((track: MediaStreamTrack) => track.readyState === 'ended'))).toBe(true)
  expect(await page.evaluate(() => JSON.stringify(sessionStorage))).not.toContain('test-voice-token')
  await page.getByRole('textbox', { name: 'Message', exact: true }).fill('Continue with text')
  await page.getByRole('button', { name: 'Send message' }).click()
  await expect(page.getByText('Text answer', { exact: true })).toHaveCount(1)
})

test('navigation disconnects and refreshes pending history on return', async ({ page }) => {
  const { calls, messages } = await setup(page)
  await mockRoom(page)
  await page.getByRole('button', { name: 'Start Voice Mode' }).click()
  await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeEnabled()
  const sessionId = calls.voice[0]
  await page.getByRole('link', { name: 'About Chatbot', exact: true }).click()
  await expect(page).toHaveURL('/about-chatbot')
  await expect.poll(() => page.evaluate(() => Reflect.get(window, 'voiceTest').stream.getTracks().every((track: MediaStreamTrack) => track.readyState === 'ended'))).toBe(true)
  messages.push({ role: 'assistant', content: 'Saved after navigation' })
  await page.getByRole('link', { name: 'Chat', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeEnabled({ timeout: 15_000 })
  await expect(page.getByText('Saved after navigation', { exact: true })).toHaveCount(1)
  await page.getByRole('button', { name: 'Start Voice Mode' }).click()
  await expect.poll(() => calls.voice.length).toBe(2)
  expect(calls.voice[1]).toBe(sessionId)
})

for (const failure of ['permission', 'connection', 'rate limit', 'conflict'] as const) {
  test(`handles ${failure} failure and restores text chat`, async ({ page }) => {
    await setup(page, { tokenStatus: failure === 'rate limit' ? 429 : failure === 'conflict' ? 409 : 200 })
    await mockRoom(page, { deny: failure === 'permission', fail: failure === 'connection' })
    await page.getByRole('button', { name: 'Start Voice Mode' }).click()
    await expect(page.getByRole('alert')).toContainText(failure === 'permission' ? 'Microphone access failed' : failure === 'rate limit' ? 'Too many voice' : failure === 'conflict' ? 'conversation is busy' : 'Could not connect')
    await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeEnabled({ timeout: 15_000 })
  })
}

test('leaving during delayed microphone access releases the eventual stream', async ({ page }) => {
  await setup(page)
  await mockRoom(page, { microphoneDelay: 700 })
  await page.getByRole('button', { name: 'Start Voice Mode' }).click()
  await expect(page.getByRole('button', { name: 'Unmute microphone', exact: true })).toBeVisible()
  await page.getByRole('link', { name: 'About Chatbot', exact: true }).click()
  await page.waitForTimeout(1000)
  expect(await page.evaluate(() => Reflect.get(window, 'voiceTest').stream.getTracks().every((track: MediaStreamTrack) => track.readyState === 'ended'))).toBe(true)
})

test('ending voice stops media before slow signaling teardown completes', async ({ page }) => {
  const { calls } = await setup(page)
  await mockRoom(page, { disconnectDelay: 2000 })
  await page.getByRole('button', { name: 'Start Voice Mode' }).click()
  await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeEnabled()
  await addVisualAgent(page)
  await expect(page.locator('audio')).toHaveCount(1)
  const before = calls.history
  await page.getByRole('button', { name: 'End conversation', exact: true }).click()
  await expect(page.locator('audio')).toHaveCount(0, { timeout: 500 })
  expect(await page.evaluate(() => Reflect.get(window, 'voiceTest').stream.getTracks().every((track: MediaStreamTrack) => track.readyState === 'ended'))).toBe(true)
  expect(calls.history).toBe(before)
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeEnabled({ timeout: 15_000 })
})

test('cancellation during room connection never enables the microphone', async ({ page }) => {
  await setup(page)
  await mockRoom(page, { connectDelay: 1200 })
  await page.getByRole('button', { name: 'Start Voice Mode' }).click()
  await expect.poll(() => page.evaluate(() => Reflect.get(window, 'voiceTest').connections)).toBe(1)
  await page.getByRole('button', { name: 'End conversation', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeEnabled({ timeout: 15_000 })
  expect(await page.evaluate(() => Reflect.get(window, 'voiceTest').stream)).toBeNull()
  await expect(page.getByRole('alert')).toHaveCount(0)
})

test('old disconnect events and delayed microphone failures cannot close a new session', async ({ page }) => {
  await setup(page)
  await mockRoom(page)
  const start = page.getByRole('button', { name: 'Start Voice Mode' })
  await start.click()
  await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeEnabled()
  await page.evaluate(() => {
    const oldRoom = Reflect.get(window, 'voiceTest').room as import('livekit-client').Room
    Reflect.set(window, 'oldVoiceRoom', oldRoom)
    oldRoom.localParticipant.setMicrophoneEnabled = () => new Promise((_resolve, reject) => Reflect.set(window, 'rejectOldMicrophone', reject))
  })
  await page.getByRole('button', { name: 'Mute microphone', exact: true }).click()
  await page.getByRole('button', { name: 'End conversation', exact: true }).click()
  await expect(start).toBeEnabled({ timeout: 15_000 })
  await start.click()
  await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeEnabled()
  await page.evaluate(() => {
    Reflect.get(window, 'oldVoiceRoom').emit('disconnected', 1)
    Reflect.get(window, 'rejectOldMicrophone')(new Error('Late device failure'))
  })
  await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeEnabled()
  await expect(page.getByRole('alert')).toHaveCount(0)
  expect(await page.evaluate(() => Reflect.get(window, 'voiceTest').room.state)).toBe('connected')
})

test('unavailable workers time out and restore normal chat', async ({ page }) => {
  await page.clock.install()
  await setup(page)
  await mockRoom(page)
  await page.getByRole('button', { name: 'Start Voice Mode' }).click()
  await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeEnabled()
  await page.clock.fastForward(30_100)
  await expect(page.getByRole('alert')).toContainText('voice assistant is unavailable')
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeEnabled({ timeout: 15_000 })
  await expect(page.getByRole('textbox', { name: 'Message', exact: true })).toBeEnabled()
})

test('pagehide releases media and terminal connection loss restores chat', async ({ page }) => {
  await setup(page)
  await mockRoom(page)
  await page.getByRole('button', { name: 'Start Voice Mode' }).click()
  await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeEnabled()
  await page.evaluate(() => window.dispatchEvent(new PageTransitionEvent('pagehide')))
  expect(await page.evaluate(() => Reflect.get(window, 'voiceTest').stream.getTracks().every((track: MediaStreamTrack) => track.readyState === 'ended'))).toBe(true)
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeEnabled({ timeout: 15_000 })
  await page.getByRole('button', { name: 'Start Voice Mode' }).click()
  await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeEnabled()
  await page.evaluate(() => {
    const room = Reflect.get(window, 'voiceTest').room as import('livekit-client').Room
    room.state = 'disconnected' as import('livekit-client').ConnectionState
    room.emit('disconnected', 2)
  })
  await expect(page.getByRole('alert')).toContainText('voice connection was lost')
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeEnabled({ timeout: 15_000 })
})

test('refresh stops the old microphone without automatically reconnecting', async ({ page }) => {
  await setup(page)
  await mockRoom(page)
  await page.getByRole('button', { name: 'Start Voice Mode' }).click()
  await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeEnabled()
  await page.evaluate(() => {
    window.addEventListener('pagehide', () => {
      sessionStorage.setItem('voice-test-page-exit', String(Reflect.get(window, 'voiceTest').stream.getTracks().every((track: MediaStreamTrack) => track.readyState === 'ended')))
    })
  })
  await page.reload()
  expect(await page.evaluate(() => sessionStorage.getItem('voice-test-page-exit'))).toBe('true')
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeEnabled()
  await expect(page.getByRole('region', { name: 'Voice Mode' })).toHaveCount(0)
  await expect(page.locator('audio')).toHaveCount(0)
})

test('live media smoke: real token endpoint, WebRTC, microphone and agent audio', async ({ page }) => {
  test.skip(process.env.VOICE_LIVE_TEST !== '1', 'Requires local API :8765 and registered voice worker; makes a provider call.')
  await page.addInitScript(() => {
    const capture = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices)
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      const stream = await capture(constraints)
      Object.assign(window, { liveVoiceStream: stream })
      return stream
    }
  })
  await page.goto('/')
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeEnabled()
  await page.getByRole('button', { name: 'Start Voice Mode' }).click()
  await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeEnabled({ timeout: 30_000 })
  await expect.poll(() => page.evaluate(() => {
    const audio = document.querySelector('audio')
    return Boolean(audio && !audio.paused && audio.currentTime > 0.2 && audio.srcObject instanceof MediaStream && audio.srcObject.getAudioTracks().some((track) => track.readyState === 'live'))
  }), { timeout: 30_000 }).toBe(true)
  await page.getByRole('button', { name: 'End conversation', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeEnabled({ timeout: 15_000 })
  expect(await page.evaluate(() => Reflect.get(window, 'liveVoiceStream').getTracks().every((track: MediaStreamTrack) => track.readyState === 'ended'))).toBe(true)
  await expect(page.locator('audio')).toHaveCount(0)
})
