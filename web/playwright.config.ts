import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './qa',
  testMatch: '*.spec.ts',
  fullyParallel: false,
  workers: 1,
  reporter: [['list'], ['json', { outputFile: 'qa/results.json' }]],
  use: {
    baseURL: 'http://127.0.0.1:5173',
    channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
    viewport: { width: 1440, height: 1000 },
    screenshot: 'only-on-failure',
  },
  webServer: { command: 'npm run dev -- --port 5173', url: 'http://127.0.0.1:5173', reuseExistingServer: true },
})
