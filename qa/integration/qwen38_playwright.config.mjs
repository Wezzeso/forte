import { defineConfig } from '../../web/node_modules/@playwright/test/index.mjs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const folder = path.dirname(fileURLToPath(import.meta.url))
export default defineConfig({
  testDir: folder,
  testMatch: 'qwen38_ui.spec.ts',
  workers: 1,
  fullyParallel: false,
  timeout: 30000,
  reporter: [['list'], ['json', { outputFile: path.join(folder, 'qwen38_ui_results.json') }]],
  outputDir: path.join(folder, 'qwen38_ui_artifacts'),
  use: { baseURL: 'http://127.0.0.1:5173', channel: 'chrome', viewport: { width: 1440, height: 1000 }, screenshot: 'only-on-failure' },
  webServer: { command: 'npm run dev -- --port 5173', cwd: path.resolve(folder, '../../web'), url: 'http://127.0.0.1:5173', reuseExistingServer: true },
})
