/** Independent synthetic integration checks. Never send private media to these fixtures. */
import { test, expect } from '../../web/node_modules/@playwright/test/index.mjs'
import type { Page } from '../../web/node_modules/@playwright/test/index.mjs'
import fs from 'node:fs'
import path from 'node:path'
import { createHash } from 'node:crypto'

const root = process.cwd()
const uiFiles = ['web/src/App.tsx', 'web/src/styles.css', 'web/src/HumanReviewPanel.tsx', 'web/src/api.ts', 'qa/integration/qwen38_ui.spec.ts']
const uiHashes = () => Object.fromEntries(uiFiles.map(name => [name, createHash('sha256').update(fs.readFileSync(path.join(root, name))).digest('hex')]))
let before: ReturnType<typeof uiHashes>
const outcomes: { title: string; status: string | undefined; expectedStatus: string; durationMs: number }[] = []
test.beforeAll(() => { before = uiHashes() })
test.afterEach(({}, info) => { outcomes.push({ title: info.title, status: info.status, expectedStatus: info.expectedStatus, durationMs: info.duration }) })
test.afterAll(() => {
  const after = uiHashes()
  fs.writeFileSync(path.join(root, 'qa/integration/qwen38_ui_aggregate.json'), JSON.stringify({
    schema_version: 'forte.qwen38.synthetic-ui-qa.v1', captured_at_utc: new Date().toISOString(),
    outcomes, source_hashes_before: before, source_hashes_after: after,
    source_changed_during_check: uiFiles.filter(name => before[name] !== after[name]),
    fixtures: 'Generated local MOV/JPEG and mocked local HTTP routes only',
    zoom_check: '200% CSS layout zoom, not browser menu zoom or full WCAG certification',
    limitations: ['No private footage, model inference, production readiness, or real upload-to-ready latency claim.'],
  }, null, 2) + '\n')
})
const fixture = JSON.parse(fs.readFileSync(path.join(root, 'contracts/fixtures/positive.json'), 'utf8'))
const media = path.join(root, 'web/qa/fixtures/review-synthetic.mov')
const key = 'synthetic-independent-qa-only'
const resultHash = 'a'.repeat(64)
const evidenceHash = (index: number) => String(index + 1).repeat(64)
const pending = (index: number) => ({ job_id: fixture.job_id, result_sha256: resultHash, violation_index: index,
  evidence_sha256: evidenceHash(index), state: 'pending', revision: 0, claimed_reviewer_name: null,
  comment: '', reviewed_at: null, reviewer_identity_verified: false })
const ledger = () => ({ job_id: fixture.job_id, result_sha256: resultHash, source_sha256: 'b'.repeat(64),
  reviewer_identity_verified: false, violations: [0, 1].map(index => ({ violation_index: index, evidence_sha256: evidenceHash(index), review: pending(index) })) })
const model = () => ({ ...fixture, violations: [
  { ...fixture.violations[0], type: 'Первое синтетическое событие', description: 'Искусственный пример проверки интерфейса. Не относится к реальной записи.' },
  { ...fixture.violations[0], type: 'Второе синтетическое событие', timecode_sec: 7.2, description: 'Другой искусственный пример. Не является правовым выводом.' },
] })

async function setup(page: Page, mode: 'positive' | 'negative' | 'failed' | 'processing' = 'positive', restore = true) {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.route('**/runtime-config.json', route => route.fulfill({ json: { apiBaseUrl: '' } }))
  await page.route('**/readyz', route => route.fulfill({ json: { status: 'ready', components: { database: true, storage: true, model: true, worker: true } } }))
  await page.route('**/api/ui/config', route => {
    expect(route.request().headers()['x-api-key']).toBe(key)
    return route.fulfill({ json: { max_upload_bytes: 256 * 1024 * 1024, accepted_suffixes: ['.mov'] } })
  })
  await page.route('**/api/v1/analyze', route => {
    expect(route.request().headers()['x-api-key']).toBe(key)
    return route.fulfill({ status: 202, json: { job_id: fixture.job_id } })
  })
  await page.route('**/api/v1/jobs/*', route => {
    expect(route.request().headers()['x-api-key']).toBe(key)
    return route.fulfill({ json: mode === 'positive' ? model() : mode === 'negative'
      ? { job_id: fixture.job_id, status: 'done', verdict: 'no_violation', violations: [] }
      : { job_id: fixture.job_id, status: mode } })
  })
  await page.route('**/api/ui/jobs/*', route => route.fulfill({ json: { job_id: fixture.job_id,
    status: mode === 'processing' ? 'processing' : mode === 'failed' ? 'failed' : 'done',
    stage: mode === 'processing' ? 'screening' : mode === 'failed' ? 'failed' : 'done',
    elapsed_sec: 12.5, limitations: ['Синтетическое ограничение видимости.'],
    events: [0, 1].map(() => ({ observed_facts: ['Искусственный видимый факт для проверки отображения.'],
      rule: { title: 'Синтетический документ', clause: 'TEST-1', url: 'https://example.com/synthetic' } })) } }))
  await page.route('**/api/ui/jobs/*/video', route => route.fulfill({ contentType: 'video/quicktime', path: media }))
  await page.route('**/api/ui/jobs/*/reviews', route => route.fulfill({ json: ledger() }))
  await page.addInitScript(({ testKey, job }) => { sessionStorage.setItem('forte.session.key', testKey); if (job) sessionStorage.setItem('forte.session.job', job) }, { testKey: key, job: restore ? fixture.job_id : '' })
  return errors
}

async function screenshot(page: Page, name: string) {
  // Full-page Chromium captures otherwise include offscreen fixed elements
  // relative to the last scrolled viewport; normalize capture position only.
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: path.join(root, 'qa/integration/qwen38_screenshots', `${name}.png`), fullPage: true })
}

for (const width of [1440, 360]) {
  test(`synthetic upload/result at ${width}px has no horizontal overflow and supports keyboard evidence`, async ({ page }) => {
    const errors = await setup(page, 'positive', false)
    await page.setViewportSize({ width, height: 1000 })
    await page.goto('/')
    await expect(page.getByRole('heading', { name: 'Всё начинается с внимания' })).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width)
    await screenshot(page, `upload-${width}`)
    await page.locator('input[type=file]').setInputFiles(media)
    await page.getByRole('button', { name: 'Начать проверку' }).click()
    await expect(page.getByRole('heading', { name: 'Обнаружены возможные признаки нарушений' })).toBeFocused()
    await expect(page.locator('[data-e2e-state]')).toHaveAttribute('data-e2e-state', 'complete')
    expect(Number(await page.locator('[data-e2e-seconds]').getAttribute('data-e2e-seconds'))).toBeGreaterThanOrEqual(0)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width)
    await expect(page.locator('.event-card')).toHaveCount(2)
    await screenshot(page, `positive-${width}`)
    await page.getByRole('button', { name: /Второе синтетическое событие/ }).focus()
    await page.keyboard.press('Enter')
    await expect(page.getByRole('button', { name: /Второе синтетическое событие/ })).toHaveAttribute('aria-pressed', 'true')
    expect(await page.locator('video').evaluate((video: HTMLVideoElement) => video.currentTime)).toBeCloseTo(7.2, 1)
    const frame = page.getByRole('button', { name: /Кадр 1/ })
    await frame.focus()
    await page.keyboard.press('Enter')
    const modal = page.getByRole('dialog', { name: /Исходный кадр/ })
    await expect(modal.getByRole('button', { name: 'Закрыть', exact: true })).toBeFocused()
    await page.keyboard.press('Shift+Tab')
    await expect(modal.getByRole('button', { name: 'К этому моменту в видео' })).toBeFocused()
    await page.keyboard.press('Tab')
    await expect(modal.getByRole('button', { name: 'Закрыть', exact: true })).toBeFocused()
    await page.keyboard.press('Escape')
    await expect(frame).toBeFocused()
    await page.reload()
    await expect(page.locator('[data-e2e-state]')).toHaveAttribute('data-e2e-state', 'restored')
    await expect(page.getByText('Замер недоступен после восстановления', { exact: true })).toBeVisible()
    await expect(page.getByText('Интервал сервера после приёма: 12,5 с.', { exact: false })).toBeVisible()
    expect(errors).toEqual([])
  })
}

test('200% CSS layout zoom reflows upload, multiple findings and review controls', async ({ page }) => {
  await setup(page, 'positive', false)
  await page.goto('/')
  await page.addStyleTag({ content: 'html { zoom: 2; }' })
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1440)
  await expect(page.getByRole('button', { name: 'Начать проверку' })).toBeVisible()
  await screenshot(page, 'upload-css-200-percent')
  await page.locator('input[type=file]').setInputFiles(media)
  await page.getByRole('button', { name: 'Начать проверку' }).click()
  await expect(page.locator('.event-card')).toHaveCount(2)
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1440)
  expect(await page.locator('.review-grid').evaluate((element: HTMLElement) => getComputedStyle(element).gridTemplateColumns.trim().split(/\s+/).length)).toBe(1)
  await page.getByLabel('Имя проверяющего', { exact: true }).fill('Синтетическое имя')
  await page.getByLabel('Комментарий', { exact: false }).fill('Искусственный текст при 200% масштабе.')
  const confirm = page.getByRole('button', { name: 'Подтвердить признак', exact: true })
  await confirm.scrollIntoViewIfNeeded()
  await expect(confirm).toBeInViewport()
  await screenshot(page, 'positive-css-200-percent')
})

for (const mode of ['negative', 'failed', 'processing'] as const) {
  test(`${mode} synthetic state preserves honest verdict and timing`, async ({ page }) => {
    const errors = await setup(page, mode)
    await page.goto('/')
    await expect(page.getByRole('heading', { name: mode === 'negative' ? 'Признаки нарушений не обнаружены' : mode === 'failed' ? 'Не удалось завершить проверку' : 'Анализируем видеоряд' })).toBeVisible()
    if (mode === 'processing') await expect(page.getByText(/%/)).toHaveCount(0)
    if (mode === 'failed') {
      await expect(page.locator('[data-e2e-state]')).toHaveAttribute('data-e2e-state', 'failed')
      await expect(page.getByRole('heading', { name: 'Признаки нарушений не обнаружены' })).toHaveCount(0)
    }
    if (mode === 'negative') await expect(page.locator('.event-card')).toHaveCount(0)
    await screenshot(page, mode)
    expect(errors).toEqual([])
  })
}

test('network polling error keeps job and offers explicit recovery', async ({ page }) => {
  await setup(page, 'processing')
  await page.route('**/api/v1/jobs/*', route => route.abort('internetdisconnected'))
  await page.goto('/')
  await expect(page.getByRole('alert')).toContainText('Не удалось обновить статус')
  await expect(page.getByRole('button', { name: 'Повторить', exact: true })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Признаки нарушений не обнаружены' })).toHaveCount(0)
  expect(await page.evaluate(() => sessionStorage.getItem('forte.session.job'))).toBe(fixture.job_id)
})

test('broken JPEG in an unselected finding prevents full client timing credit', async ({ page }) => {
  await setup(page, 'positive', false)
  const result = model()
  result.violations[1].evidence_frames = [{ timestamp_sec: 7.2, image_base64: '/9j/synthetic-invalid-jpeg' }]
  await page.route('**/api/v1/jobs/*', route => route.fulfill({ json: result }))
  await page.goto('/')
  await page.locator('input[type=file]').setInputFiles(media)
  await page.getByRole('button', { name: 'Начать проверку' }).click()
  await expect(page.getByRole('heading', { name: 'Обнаружены возможные признаки нарушений' })).toBeVisible()
  await expect(page.locator('[data-e2e-state]')).toHaveAttribute('data-e2e-state', 'incomplete')
  await expect(page.locator('[data-e2e-seconds]')).toHaveCount(0)
  await expect(page.getByRole('alert')).toContainText('Не все исходные JPEG-кадры удалось открыть')
  await expect(page.getByRole('button', { name: 'Повторить получение результата', exact: true })).toBeVisible()
})

test('source decode failure prevents a completed client timing claim', async ({ page }) => {
  await setup(page, 'positive', false)
  await page.goto('/')
  await page.locator('input[type=file]').setInputFiles({ name: 'synthetic-corrupt.mov', mimeType: 'video/quicktime', buffer: Buffer.from('synthetic corrupt container only') })
  await page.getByRole('button', { name: 'Начать проверку' }).click()
  await expect(page.getByRole('heading', { name: 'Обнаружены возможные признаки нарушений' })).toBeVisible()
  await expect(page.locator('[data-e2e-state]')).toHaveAttribute('data-e2e-state', 'incomplete')
  await expect(page.locator('[data-e2e-seconds]')).toHaveCount(0)
  await expect(page.getByText('Браузер не смог воспроизвести этот кодек MOV.', { exact: false })).toBeVisible()
})

test('authenticated configured upload cap is displayed and blocks oversized synthetic file', async ({ page }) => {
  await setup(page, 'positive', false)
  let uploads = 0
  await page.route('**/api/ui/config', route => {
    expect(route.request().headers()['x-api-key']).toBe(key)
    return route.fulfill({ json: { max_upload_bytes: 1024 * 1024, accepted_suffixes: ['.mov'] } })
  })
  page.on('request', request => { if (new URL(request.url()).pathname === '/api/v1/analyze') uploads++ })
  await page.goto('/')
  await expect(page.locator('[data-max-upload-bytes]')).toHaveAttribute('data-max-upload-bytes', '1048576')
  await expect(page.getByText('Лимит сервера: 1 МБ.', { exact: false })).toBeVisible()
  await page.locator('input[type=file]').setInputFiles({ name: 'synthetic-oversized.mov', mimeType: 'video/quicktime', buffer: Buffer.alloc(1024 * 1024 + 1) })
  await expect(page.getByRole('alert')).toContainText('превышает лимит сервера')
  await expect(page.getByRole('button', { name: 'Начать проверку' })).toBeDisabled()
  expect(uploads).toBe(0)
})

test('unavailable upload cap is disclosed without inventing a configured value', async ({ page }) => {
  await setup(page, 'positive', false)
  await page.route('**/api/ui/config', route => route.fulfill({ status: 503, json: { detail: 'synthetic_unavailable' } }))
  await page.goto('/')
  await expect(page.getByText('Текущий лимит размера не подтверждён.', { exact: false })).toBeVisible()
  await expect(page.locator('[data-max-upload-bytes]')).toHaveCount(0)
  await page.locator('input[type=file]').setInputFiles(media)
  await expect(page.getByRole('button', { name: 'Начать проверку' })).toBeEnabled()
})

test('review conflict preserves drafts and frozen full client timing before explicit retry', async ({ page }) => {
  await setup(page, 'positive', false)
  const state = ledger()
  let attempts = 0
  await page.route('**/api/ui/jobs/*/reviews', route => route.fulfill({ json: state }))
  await page.route(/\/api\/ui\/jobs\/[^/]+\/reviews\/0$/, route => {
    attempts++
    const input = route.request().postDataJSON()
    if (attempts === 1) {
      state.violations[0].review = { ...pending(0), revision: 1, state: 'rejected', claimed_reviewer_name: 'Другой искусственный пользователь', comment: 'Ранее сохранённый искусственный текст.', reviewed_at: '2026-10-17T12:00:00Z' }
      return route.fulfill({ status: 409, json: { detail: 'review_conflict', current_revision: 1, current_result_sha256: resultHash } })
    }
    expect(input.expected_revision).toBe(1)
    expect(input.comment).toBe('Искусственный черновик сохраняется.')
    expect(input.evidence_sha256).toBe(evidenceHash(0))
    return route.fulfill({ json: { ...pending(0), state: 'confirmed', revision: 2, claimed_reviewer_name: input.claimed_reviewer_name, comment: input.comment, reviewed_at: '2026-10-17T12:01:00Z' } })
  })
  await page.goto('/')
  await page.locator('input[type=file]').setInputFiles(media)
  await page.getByRole('button', { name: 'Начать проверку' }).click()
  const clock = page.locator('[data-e2e-seconds]')
  await expect(clock).toHaveAttribute('data-e2e-state', 'complete')
  const frozenSeconds = await clock.getAttribute('data-e2e-seconds')
  await page.getByLabel('Имя проверяющего', { exact: true }).fill('Искусственный черновик')
  await page.getByLabel('Комментарий', { exact: false }).fill('Искусственный черновик сохраняется.')
  await page.getByRole('button', { name: 'Подтвердить признак', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('Конфликт версий')
  await expect(page.getByLabel('Комментарий', { exact: false })).toHaveValue('Искусственный черновик сохраняется.')
  await expect(clock).toHaveAttribute('data-e2e-seconds', frozenSeconds!)
  await screenshot(page, 'human-conflict')
  await page.getByRole('button', { name: 'Подтвердить признак', exact: true }).click()
  await expect(page.getByText('Решение сохранено. Версия 2.', { exact: true })).toBeVisible()
  await expect(clock).toHaveAttribute('data-e2e-seconds', frozenSeconds!)
  expect(attempts).toBe(2)
})
