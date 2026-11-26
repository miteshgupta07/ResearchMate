import { defineConfig } from '@playwright/test'
import { mkdirSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const silentAudio = resolve('node_modules/.cache/voice-silence.wav')
if (process.env.VOICE_LIVE_TEST === '1') {
  // Synthetic silence exercises media capture without sending test speech to history.
  const wav = Buffer.alloc(44 + 96000)
  wav.write('RIFF', 0); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8)
  wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22)
  wav.writeUInt32LE(48000, 24); wav.writeUInt32LE(96000, 28)
  wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34)
  wav.write('data', 36); wav.writeUInt32LE(96000, 40)
  mkdirSync(resolve('node_modules/.cache'), { recursive: true })
  writeFileSync(silentAudio, wav)
}

export default defineConfig({
  testDir: './tests',
  timeout: 45_000,
  workers: 2,
  outputDir: './node_modules/.cache/playwright-results',
  use: {
    baseURL: 'http://127.0.0.1:4173',
    headless: true,
    launchOptions: { args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--autoplay-policy=no-user-gesture-required', ...(process.env.VOICE_LIVE_TEST === '1' ? [`--use-file-for-fake-audio-capture=${silentAudio}`] : [])] },
  },
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 4173 --strictPort',
    url: 'http://127.0.0.1:4173',
    env: { VITE_API_BASE_URL: 'http://127.0.0.1:8765' },
    timeout: 30_000,
  },
})
