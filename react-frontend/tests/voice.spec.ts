import { expect, test } from '@playwright/test'
import type { Page } from '@playwright/test'

type Message = { role: string; content: string }

async function setup(page: Page, options: { historyDelay?: number; tokenDelay?: number; tokenStatus?: number; textDelay?: number; uploadDelay?: number } = {}) {
  const calls: { history: number; voice: string[]; chat: string[] } = { history: 0, voice: [], chat: [] }
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
      messages.push({ role: 'user', content: data.message }, { role: 'assistant', content: 'Text answer' })
      body = { role: 'assistant', content: 'Text answer', sources: [], intent: 'INTENT_CHAT', confidence: 1, metadata: {} }
    }
    await route.fulfill({ status, headers, contentType: 'application/json', body: JSON.stringify(body) })
  })
  await page.goto('/')
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeVisible()
  return { calls, messages }
}

/** Keep real LiveKit React hooks/audio renderer; stub signaling and publishing only. */
async function mockRoom(page: Page, options: { deny?: boolean; fail?: boolean; microphoneDelay?: number } = {}) {
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
        if (enabled && !state.stream) state.stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        if (this.state === sdk.ConnectionState.Disconnected) {
          state.stream?.getTracks().forEach((track) => track.stop())
          throw new Error('Disconnected during device access')
        }
        state.mic = enabled
        this.localParticipant.emit(sdk.ParticipantEvent.ParticipantPermissionsChanged)
        return undefined
      }
      this.state = sdk.ConnectionState.Connected
      this.emit(sdk.RoomEvent.ConnectionStateChanged, sdk.ConnectionState.Connected)
      this.emit(sdk.RoomEvent.Connected)
    }
    const disconnect = async function (this: import('livekit-client').Room) {
      if (this.state === sdk.ConnectionState.Disconnected) return
      state.disconnects += 1
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
  const { calls, messages } = await setup(page)
  await mockRoom(page)
  await page.getByRole('button', { name: 'Start Voice Mode' }).click()
  await expect(page.getByRole('button', { name: 'Mute microphone', exact: true })).toBeEnabled()
  await page.getByRole('button', { name: 'Mute microphone', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Unmute microphone', exact: true })).toBeEnabled()
  await page.getByRole('button', { name: 'Unmute microphone', exact: true }).click()
  const historyBefore = calls.history
  await page.getByRole('button', { name: 'End conversation', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeDisabled()
  await page.waitForTimeout(1800)
  messages.push({ role: 'user', content: 'Voice question' }, { role: 'assistant', content: 'Voice answer' })
  await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeEnabled({ timeout: 15_000 })
  await expect(page.getByText('Voice answer', { exact: true })).toHaveCount(1)
  await expect(page.getByText('Voice question', { exact: true })).toHaveCount(1)
  expect(calls.history - historyBefore).toBe(5)
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

for (const failure of ['permission', 'connection', 'rate limit'] as const) {
  test(`handles ${failure} failure and restores text chat`, async ({ page }) => {
    await setup(page, { tokenStatus: failure === 'rate limit' ? 429 : 200 })
    await mockRoom(page, { deny: failure === 'permission', fail: failure === 'connection' })
    await page.getByRole('button', { name: 'Start Voice Mode' }).click()
    await expect(page.getByRole('alert')).toContainText(failure === 'permission' ? 'Microphone access failed' : failure === 'rate limit' ? 'Too many voice' : 'Could not connect')
    await expect(page.getByRole('button', { name: 'Start Voice Mode' })).toBeEnabled({ timeout: 15_000 })
  })
}

test('leaving during delayed microphone access releases the eventual stream', async ({ page }) => {
  await setup(page)
  await mockRoom(page, { microphoneDelay: 700 })
  await page.getByRole('button', { name: 'Start Voice Mode' }).click()
  await expect(page.getByText('Connected · Microphone muted', { exact: true })).toBeVisible()
  await page.getByRole('link', { name: 'About Chatbot', exact: true }).click()
  await page.waitForTimeout(1000)
  expect(await page.evaluate(() => Reflect.get(window, 'voiceTest').stream.getTracks().every((track: MediaStreamTrack) => track.readyState === 'ended'))).toBe(true)
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
