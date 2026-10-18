/** Synthetic browser verification only. No test route, key, fixture, or mock is included in src/ or dist/. */
import { test, expect } from '@playwright/test'
import type { Page } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'

const positive = JSON.parse(fs.readFileSync(path.resolve('../contracts/fixtures/positive.json'), 'utf8'))
const syntheticVideo = path.resolve('qa/fixtures/review-synthetic.mov')
const screenshot = (name: string) => path.resolve(`qa/screenshots/${name}.png`)
const testKey = 'synthetic-browser-only-key'
const resultHash = 'a'.repeat(64)
const sourceHash = 'b'.repeat(64)
const evidenceHash = (index: number) => String(index + 1).repeat(64)
const pendingReview = (index: number) => ({ job_id: positive.job_id, result_sha256: resultHash, violation_index: index, evidence_sha256: evidenceHash(index), state: 'pending', revision: 0, claimed_reviewer_name: null as string | null, comment: '', reviewed_at: null as string | null, reviewer_identity_verified: false })
const pendingReviews = () => ({ job_id: positive.job_id, result_sha256: resultHash, source_sha256: sourceHash, reviewer_identity_verified: false, violations: [0, 1].map(index => ({ violation_index: index, evidence_sha256: evidenceHash(index), review: pendingReview(index) })) })
function syntheticPositive() {
  const base = positive.violations[0]
  return { ...positive, violations: [
    { ...base, type: 'Синтетическое событие для проверки интерфейса', description: 'Это искусственный пример интерфейса, а не наблюдение из реальной записи. Синтетический документ, раздел TEST-1. Не является нормативным или дисциплинарным выводом.' },
    { ...base, type: 'Второе отдельное синтетическое событие', timecode_sec: 7.2 },
  ] }
}

async function routes(page: Page, mode: 'positive' | 'negative' | 'failed' | 'processing' = 'positive') {
  await page.route('**/runtime-config.json', route => route.fulfill({ status: 404, json: { detail: 'optional_config_missing' } }))
  await page.route('**/readyz', route => route.fulfill({ json: { status: 'ready', components: { database: true, storage: true, model: true, worker: true } } }))
  await page.route('**/api/ui/config', route => {
    expect(route.request().headers()['x-api-key']).toBe(testKey)
    return route.fulfill({ json: { max_upload_bytes: 256 * 1024 * 1024, accepted_suffixes: ['.mov'] } })
  })
  await page.route('**/api/v1/analyze', async route => {
    expect(route.request().headers()['x-api-key']).toBe(testKey)
    expect(route.request().postDataBuffer()?.toString('utf8', 0, 300)).toContain('name="file"')
    await route.fulfill({ status: 202, json: { job_id: positive.job_id } })
  })
  await page.route('**/api/v1/jobs/*', async route => {
    expect(route.request().headers()['x-api-key']).toBe(testKey)
    if (mode === 'positive') {
      return route.fulfill({ json: syntheticPositive() })
    }
    return route.fulfill({ json: mode === 'negative' ? { job_id: positive.job_id, status: 'done', verdict: 'no_violation', violations: [] } : { job_id: positive.job_id, status: mode } })
  })
  await page.route('**/api/ui/jobs/*', route => route.fulfill({ json: {
    job_id: positive.job_id, status: mode === 'processing' ? 'processing' : mode === 'failed' ? 'failed' : 'done',
    stage: mode === 'processing' ? 'screening' : mode === 'failed' ? 'failed' : 'done',
    created_at: new Date(Date.now() - 15_000).toISOString(), elapsed_sec: 12.5,
    limitations: mode === 'positive' ? ['Синтетический пример ограничения видимости; реальные записи не используются.'] : [],
    events: [{ rule: { title: 'Синтетический документ для проверки источников', clause: 'Раздел TEST-1 · пример, не нормативный акт', url: 'https://example.com/synthetic' } }],
    error_code: mode === 'failed' ? 'synthetic_test_failure' : null,
  } }))
  await page.route('**/api/ui/jobs/*/video', route => route.fulfill({ contentType: 'video/quicktime', path: syntheticVideo }))
  await page.route('**/api/ui/jobs/*/reviews', route => route.fulfill({ json: pendingReviews() }))
}

async function restoreSyntheticJob(page: Page) {
  await page.addInitScript(({ key, job }) => { sessionStorage.setItem('forte.session.key', key); sessionStorage.setItem('forte.session.job', job) }, { key: testKey, job: positive.job_id })
}

test('portrait footage and evidence retain their proportions across desktop, tablet and mobile', async ({ page }) => {
  await routes(page)
  const portrait = structuredClone(syntheticPositive())
  const jpeg = fs.readFileSync(path.resolve('qa/fixtures/review-portrait-synthetic.jpg')).toString('base64')
  portrait.violations.forEach(event => event.evidence_frames.forEach((frame: { image_base64: string }) => { frame.image_base64 = jpeg }))
  await page.route('**/api/v1/jobs/*', route => route.fulfill({ json: portrait }))
  await page.route('**/api/ui/jobs/*/video', route => route.fulfill({ contentType: 'video/quicktime', path: path.resolve('qa/fixtures/review-portrait-synthetic.mov') }))
  await restoreSyntheticJob(page)
  await page.goto('/')
  const video = page.locator('.video-surface video')
  await expect.poll(() => video.evaluate((element: HTMLVideoElement) => element.videoHeight)).toBe(320)
  await expect.poll(() => video.evaluate((element: HTMLVideoElement) => element.readyState)).toBeGreaterThanOrEqual(2)
  await expect(page.locator('.review-grid')).toHaveClass(/portrait-review/)
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 1280, height: 720 }, { width: 900, height: 900 }, { width: 360, height: 800 }]) {
    await page.setViewportSize(viewport)
    await page.evaluate(() => window.scrollTo(0, 0))
    const bounds = await page.locator('.video-surface').boundingBox()
    expect(bounds!.width / bounds!.height).toBeCloseTo(9 / 16, 2)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(viewport.width)
    await expect(video).toHaveCSS('object-fit', 'contain')
    if (viewport.width >= 1024) await expect(page.locator('.video-surface')).toBeInViewport({ ratio: 1 })
    await page.screenshot({ path: screenshot(`portrait-${viewport.width}`), fullPage: true })
  }
  await page.getByRole('button', { name: /Второе отдельное синтетическое событие/ }).click()
  expect(await video.evaluate((element: HTMLVideoElement) => element.currentTime)).toBeCloseTo(7.2, 1)
  await page.getByRole('button', { name: /Кадр 1/ }).click()
  const image = page.locator('.enlarged-evidence')
  await expect.poll(() => image.evaluate((element: HTMLImageElement) => element.naturalHeight)).toBe(320)
  await expect(image).toHaveCSS('object-fit', 'contain')
  await page.keyboard.press('Escape')
  await expect(page.getByRole('dialog')).toHaveCount(0)
})

async function connect(page: Page) {
  const mobileMenu = page.getByRole('button', { name: 'Открыть меню', exact: true })
  if (await mobileMenu.isVisible()) await mobileMenu.click()
  await page.getByRole('button', { name: 'Ключ доступа', exact: true }).click()
  const field = page.getByLabel('Ключ доступа', { exact: true })
  await field.pressSequentially(testKey, { delay: 10 })
  await expect(field).toHaveValue(testKey)
  await expect(field).toBeFocused()
  await page.getByRole('button', { name: 'Подключить', exact: true }).click()
}

async function upload(page: Page) {
  await page.locator('input[type=file]').setInputFiles(syntheticVideo)
  await page.getByRole('button', { name: 'Начать проверку' }).click()
}

for (const { name, viewport, reducedMotion } of [
  { name: 'desktop', viewport: { width: 1440, height: 1000 }, reducedMotion: 'no-preference' as const },
  { name: 'mobile', viewport: { width: 390, height: 844 }, reducedMotion: 'no-preference' as const },
  { name: 'reduced motion', viewport: { width: 1280, height: 720 }, reducedMotion: 'reduce' as const },
]) {
  test(`starting analysis from a scrolled upload opens the top of processing on ${name}`, async ({ page }) => {
    await page.setViewportSize(viewport)
    await page.emulateMedia({ reducedMotion })
    await routes(page, 'processing')
    let acceptUpload!: () => void
    const accepted = new Promise<void>(resolve => { acceptUpload = resolve })
    await page.route('**/api/v1/analyze', async route => {
      await accepted
      await route.fulfill({ status: 202, json: { job_id: positive.job_id } })
    })
    await page.goto('/')
    await connect(page)
    await page.locator('input[type=file]').setInputFiles(syntheticVideo)
    await page.getByRole('link', { name: 'Проверить видео', exact: true }).click()
    await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(200)
    await page.getByRole('button', { name: 'Начать проверку', exact: true }).click()
    await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(0)
    await expect(page.getByRole('heading', { name: 'Проверяем ваше видео', exact: true })).toBeInViewport()
    // Receiving a job ID and polling updates must not pull an expert back up.
    await page.mouse.wheel(0, 220)
    await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(80)
    acceptUpload()
    await expect(page.getByRole('heading', { name: 'Анализируем видеоряд', exact: true })).toBeVisible()
    expect(await page.evaluate(() => window.scrollY)).toBeGreaterThan(80)
  })
}

test('upload, runtime key typing, real request shape, two events, source seeking, evidence and visibility', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await routes(page)
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Видим суть. Решаете вы.' })).toBeVisible()
  await expect(page).toHaveTitle('FORTE Panopticon — Видеоаналитика')
  expect(await page.evaluate(async () => { const faces = await document.fonts.load('16px "UpMax"'); return faces.length > 0 && faces.every(face => face.status === 'loaded') })).toBe(true)
  await expect(page.getByRole('button', { name: 'Начать проверку' })).toBeDisabled()
  await page.screenshot({ path: screenshot('upload-desktop'), fullPage: true })
  await connect(page)
  await upload(page)
  await expect(page.getByRole('heading', { name: 'Обнаружены возможные признаки нарушений' })).toBeVisible()
  await expect(page).toHaveTitle('Результат проверки — FORTE Panopticon')
  await expect(page.locator('.result-diagnostics')).toHaveJSProperty('open', false)
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(0)
  for (const selector of ['.video-surface', '.events-panel']) {
    const area = await page.locator(selector).boundingBox()
    expect(area, `${selector} is rendered`).not.toBeNull()
    expect(area!.y).toBeGreaterThanOrEqual(0)
    expect(area!.y, `${selector} leaves useful review content visible above the fold`).toBeLessThan(400)
  }
  await expect(page.locator('.video-surface')).toBeInViewport({ ratio: 1 })
  await expect(page.getByRole('button', { name: /Второе отдельное синтетическое событие/ })).toBeVisible()
  await expect(page.getByText('Синтетический пример ограничения видимости; реальные записи не используются.')).toBeVisible()
  await page.locator('video').evaluate((video: HTMLVideoElement) => new Promise<void>(resolve => { if (video.readyState >= 1) resolve(); else video.addEventListener('loadedmetadata', () => resolve(), { once: true }) }))
  await page.getByRole('button', { name: /Второе отдельное синтетическое событие/ }).click()
  expect(await page.locator('video').evaluate((video: HTMLVideoElement) => video.currentTime)).toBeCloseTo(7.2, 1)
  await page.getByRole('button', { name: /Синтетическое событие для проверки интерфейса/ }).click()
  await page.screenshot({ path: screenshot('positive-multiple-desktop'), fullPage: true })
  await page.setViewportSize({ width: 1920, height: 1000 })
  const footerWidth = await page.locator('.page-footer').evaluate(footer => {
    const bounds = footer.getBoundingClientRect()
    return { left: bounds.left, right: bounds.right, viewport: document.documentElement.clientWidth }
  })
  expect(Math.abs(footerWidth.left)).toBeLessThanOrEqual(1)
  expect(Math.abs(footerWidth.right - footerWidth.viewport)).toBeLessThanOrEqual(1)
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(1920)
  await page.setViewportSize({ width: 1280, height: 720 })
  await expect(page.locator('.video-surface')).toBeInViewport({ ratio: 1 })
  await page.setViewportSize({ width: 1440, height: 1000 })
  await page.getByRole('button', { name: /Кадр 1/ }).click()
  const dialog = page.getByRole('dialog', { name: /Исходный кадр/ })
  await expect(dialog).toBeVisible()
  await expect(dialog.getByRole('button', { name: 'Закрыть', exact: true })).toBeFocused()
  await page.keyboard.press('Shift+Tab')
  await expect(dialog.getByRole('button', { name: 'К этому моменту в видео' })).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(dialog.getByRole('button', { name: 'Закрыть', exact: true })).toBeFocused()
  await page.screenshot({ path: screenshot('evidence-modal'), fullPage: false })
  await page.keyboard.press('Escape')
  await expect(page.getByRole('dialog')).toHaveCount(0)
  await expect(page.getByRole('button', { name: /Кадр 1/ })).toBeFocused()
  expect(errors).toEqual([])
})

for (const mode of ['negative', 'failed', 'processing'] as const) {
  test(`${mode} state remains honest`, async ({ page }) => {
    await routes(page, mode)
    await page.addInitScript(({ key, job }) => { sessionStorage.setItem('forte.session.key', key); sessionStorage.setItem('forte.session.job', job) }, { key: testKey, job: positive.job_id })
    await page.goto('/')
    await expect(page.getByRole('heading', { name: mode === 'negative' ? 'Признаки нарушений не обнаружены' : mode === 'failed' ? 'Не удалось завершить проверку' : 'Анализируем видеоряд' })).toBeVisible()
    await expect(page).toHaveTitle(mode === 'failed' ? 'Проверка не завершена — FORTE Panopticon' : mode === 'processing' ? 'Проверяем видео — FORTE Panopticon' : 'Результат проверки — FORTE Panopticon')
    if (mode !== 'processing') await expect(page.locator('.result-diagnostics')).toHaveJSProperty('open', false)
    if (mode === 'processing') { await expect(page.getByText(/%/)).toHaveCount(0); await expect(page.getByText('Этап сервиса: screening')).toBeVisible() }
    await page.screenshot({ path: screenshot(`${mode}-desktop`), fullPage: true })
  })
}

test('tablet navigation contains keyboard focus and Escape returns it to the menu toggle', async ({ page }) => {
  await routes(page)
  await page.setViewportSize({ width: 900, height: 900 })
  await page.goto('/')
  const open = page.getByRole('button', { name: 'Открыть меню', exact: true })
  await expect(open).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(900)
  await open.click()
  const navigation = page.getByRole('navigation', { name: 'Мобильная навигация', exact: true })
  const close = page.getByRole('button', { name: 'Закрыть меню', exact: true })
  const access = navigation.getByRole('button', { name: 'Ключ доступа', exact: true })
  await expect(navigation.getByRole('link', { name: 'Обзор', exact: true })).toBeFocused()
  await page.keyboard.press('Shift+Tab')
  await expect(close).toBeFocused()
  await page.keyboard.press('Shift+Tab')
  await expect(access).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(close).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(navigation.getByRole('link', { name: 'Обзор', exact: true })).toBeFocused()
  await page.keyboard.press('Escape')
  await expect(navigation).toHaveCount(0)
  await expect(open).toBeFocused()
})

test('resizing an open tablet menu to desktop restores page interaction and scrolling', async ({ page }) => {
  await routes(page)
  await page.setViewportSize({ width: 900, height: 900 })
  await page.goto('/')
  await page.getByRole('button', { name: 'Открыть меню', exact: true }).click()
  await expect(page.locator('main')).toHaveJSProperty('inert', true)
  await page.setViewportSize({ width: 1440, height: 900 })
  await expect(page.getByRole('navigation', { name: 'Мобильная навигация', exact: true })).toHaveCount(0)
  await expect(page.locator('main')).toHaveJSProperty('inert', false)
  await expect.poll(() => page.evaluate(() => document.body.style.overflow)).not.toBe('hidden')
  await page.getByRole('link', { name: 'Проверить видео', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Добавьте видео', exact: true })).toBeInViewport()
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(0)
  await page.getByText('Требования к видео', { exact: true }).click()
  await expect(page.locator('.file-constraints')).toBeVisible()
})

test('a short mobile menu scrolls to its access control and opens the key dialog', async ({ page }) => {
  await routes(page)
  await page.setViewportSize({ width: 390, height: 390 })
  await page.goto('/')
  await page.getByRole('button', { name: 'Открыть меню', exact: true }).click()
  const navigation = page.getByRole('navigation', { name: 'Мобильная навигация', exact: true })
  await navigation.hover({ position: { x: 20, y: 200 } })
  await page.mouse.wheel(0, 600)
  await expect.poll(() => navigation.evaluate(element => element.scrollTop)).toBeGreaterThan(0)
  const access = navigation.getByRole('button', { name: 'Ключ доступа', exact: true })
  await expect(access).toBeInViewport()
  await access.click()
  await expect(navigation).toHaveCount(0)
  const dialog = page.getByRole('dialog', { name: 'Подключить доступ', exact: true })
  await expect(dialog).toBeVisible()
  await expect(dialog.getByLabel('Ключ доступа', { exact: true })).toBeEditable()
  await page.keyboard.press('Escape')
  await expect(dialog).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Открыть меню', exact: true })).toBeFocused()
})

test('360px viewport and invalid file remain usable', async ({ page }) => {
  await routes(page)
  await page.setViewportSize({ width: 360, height: 900 })
  await page.goto('/')
  await expect(page.getByRole('button', { name: 'Начать проверку' })).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(360)
  await page.screenshot({ path: screenshot('upload-360'), fullPage: true })
  await page.locator('input[type=file]').setInputFiles({ name: 'synthetic.txt', mimeType: 'text/plain', buffer: Buffer.from('synthetic only') })
  await expect(page.getByRole('alert')).toContainText('Выберите видео в формате .mov.')
  await connect(page)
  await upload(page)
  await expect(page.getByRole('heading', { name: 'Обнаружены возможные признаки нарушений' })).toBeVisible()
  await expect(page.locator('.result-diagnostics')).toHaveJSProperty('open', false)
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(360)
  await page.getByText('Технические сведения', { exact: true }).click()
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(360)
  await page.getByText('Технические сведения', { exact: true }).click()
  await page.screenshot({ path: screenshot('positive-multiple-360'), fullPage: true })
})

test('lost connection keeps job and allows retry without false verdict', async ({ page }) => {
  await routes(page, 'processing')
  await page.route('**/api/v1/jobs/*', route => route.abort('internetdisconnected'))
  await page.addInitScript(({ key, job }) => { sessionStorage.setItem('forte.session.key', key); sessionStorage.setItem('forte.session.job', job) }, { key: testKey, job: positive.job_id })
  await page.goto('/')
  await expect(page.getByRole('alert')).toContainText('Не удалось обновить статус')
  await expect(page.getByRole('button', { name: 'Повторить', exact: true })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Признаки нарушений не обнаружены' })).toHaveCount(0)
  expect(await page.evaluate(() => sessionStorage.getItem('forte.session.job'))).toBe(positive.job_id)
})

test('missing restored job offers a new upload without reporting a verdict', async ({ page }) => {
  await routes(page, 'processing')
  await page.route('**/api/v1/jobs/*', route => route.fulfill({ status: 404, json: { detail: 'not_found' } }))
  await page.addInitScript(({ key, job }) => { sessionStorage.setItem('forte.session.key', key); sessionStorage.setItem('forte.session.job', job) }, { key: testKey, job: positive.job_id })
  await page.goto('/')
  await expect(page.getByRole('alert')).toContainText('Задача не найдена')
  await page.getByRole('button', { name: 'К загрузке' }).click()
  await expect(page.getByRole('heading', { name: 'Видим суть. Решаете вы.' })).toBeVisible()
  expect(await page.evaluate(() => sessionStorage.getItem('forte.session.job'))).toBe(null)
})

test('explicit service preparation blocks upload until the next real ready response', async ({ page }) => {
  await page.clock.install()
  await routes(page)
  let ready = false
  let readinessRequests = 0
  let uploadRequests = 0
  page.on('request', request => { if (request.url().endsWith('/api/v1/analyze')) uploadRequests += 1 })
  await page.route('**/readyz', route => {
    readinessRequests += 1
    return route.fulfill({ status: ready ? 200 : 503, json: { status: ready ? 'ready' : 'not_ready', components: { database: true, storage: true, model: ready, worker: ready } } })
  })
  await page.goto('/')
  await page.locator('input[type=file]').setInputFiles(syntheticVideo)
  const start = page.getByRole('button', { name: 'Начать проверку' })
  await expect(page.getByRole('status')).toContainText('Сервис готовится. Проверка станет доступна после подготовки.')
  await expect(start).toBeDisabled()
  expect(uploadRequests).toBe(0)
  ready = true
  await page.clock.fastForward(30_000)
  await expect(start).toBeEnabled()
  await expect(page.locator('#service-preparing')).toHaveCount(0)
  expect(readinessRequests).toBeGreaterThanOrEqual(2)
})

test('operator HTTPS runtime origin carries upload, polling, metadata, source video and readiness', async ({ page }) => {
  await routes(page)
  const configuredOrigin = 'https://configured-api.example'
  await page.route('**/runtime-config.json', route => route.fulfill({ json: { apiBaseUrl: configuredOrigin } }))
  const requests: { url: URL; key?: string }[] = []
  page.on('request', request => {
    const url = new URL(request.url())
    if (url.pathname.startsWith('/api/') || url.pathname === '/readyz') requests.push({ url, key: request.headers()['x-api-key'] })
  })
  await page.goto('/')
  await connect(page)
  await upload(page)
  await expect(page.getByRole('heading', { name: 'Обнаружены возможные признаки нарушений' })).toBeVisible()
  await page.reload()
  await expect(page.getByRole('heading', { name: 'Обнаружены возможные признаки нарушений' })).toBeVisible()
  await expect.poll(() => requests.some(request => request.url.pathname.endsWith('/video'))).toBe(true)
  await page.route(`${configuredOrigin}/api/ui/jobs/${positive.job_id}/reviews/0`, route => {
    const input = route.request().postDataJSON()
    expect(input.result_sha256).toBe(resultHash)
    expect(route.request().headers()['x-api-key']).toBe(testKey)
    return route.fulfill({ json: { ...pendingReview(0), state: 'confirmed', revision: 1, claimed_reviewer_name: input.claimed_reviewer_name, comment: input.comment, reviewed_at: '2026-10-17T12:00:00Z' } })
  })
  await expect(page.getByText('Ожидает решения человека', { exact: true })).toBeVisible()
  await page.getByLabel('Имя проверяющего', { exact: true }).fill('Синтетический API origin')
  await page.getByRole('button', { name: 'Подтвердить признак', exact: true }).click()
  await expect(page.getByText('Решение сохранено. Версия 1.', { exact: true })).toBeVisible()
  expect(requests.length).toBeGreaterThan(4)
  expect(requests.every(request => request.url.origin === configuredOrigin)).toBe(true)
  expect(requests.filter(request => request.url.pathname.startsWith('/api/')).every(request => request.key === testKey)).toBe(true)
  expect(requests.some(request => request.url.pathname === '/api/v1/analyze')).toBe(true)
  expect(requests.some(request => request.url.pathname === `/api/v1/jobs/${positive.job_id}`)).toBe(true)
  expect(requests.some(request => request.url.pathname === `/api/ui/jobs/${positive.job_id}`)).toBe(true)
  expect(requests.some(request => request.url.pathname === '/readyz')).toBe(true)
})

for (const scenario of ['missing', 'unsafe'] as const) {
  test(`${scenario} runtime config keeps API same-origin and ignores URL overrides`, async ({ page }) => {
    await routes(page)
    if (scenario === 'unsafe') await page.route('**/runtime-config.json', route => route.fulfill({ json: { apiBaseUrl: 'http://untrusted.example/path?token=ignored' } }))
    const requests: URL[] = []
    page.on('request', request => { const url = new URL(request.url()); if (url.pathname.startsWith('/api/') || url.pathname === '/readyz') requests.push(url) })
    await page.goto('/?apiBaseUrl=https://untrusted.example')
    await connect(page)
    await upload(page)
    await expect(page.getByRole('heading', { name: 'Обнаружены возможные признаки нарушений' })).toBeVisible()
    const frontendOrigin = new URL(page.url()).origin
    expect(requests.length).toBeGreaterThan(3)
    expect(requests.every(url => url.origin === frontendOrigin)).toBe(true)
  })
}

test('human decisions persist per finding with history while model JSON bytes and analysis time remain unchanged', async ({ page }) => {
  await routes(page)
  const canonicalJson = `\n${JSON.stringify(syntheticPositive(), null, 2)}\n`
  await page.route('**/api/v1/jobs/*', route => route.fulfill({ contentType: 'application/json', body: canonicalJson }))
  const ledger = pendingReviews()
  const history: ReturnType<typeof pendingReview>[] = []
  await page.route('**/api/ui/jobs/*/reviews', route => route.fulfill({ json: ledger }))
  await page.route(/\/api\/ui\/jobs\/[^/]+\/reviews\/\d+$/, route => {
    expect(route.request().method()).toBe('POST')
    expect(route.request().headers()['x-api-key']).toBe(testKey)
    const input = route.request().postDataJSON()
    const index = Number(new URL(route.request().url()).pathname.split('/').at(-1))
    const item = ledger.violations[index]
    expect(input.result_sha256).toBe(resultHash)
    expect(input.evidence_sha256).toBe(item.evidence_sha256)
    expect(input.expected_revision).toBe(item.review.revision)
    item.review = { ...item.review, state: input.decision === 'confirm' ? 'confirmed' : 'rejected', revision: input.expected_revision + 1, claimed_reviewer_name: input.claimed_reviewer_name, comment: input.comment, reviewed_at: '2026-10-17T12:00:00Z' }
    history.push({ ...item.review })
    return route.fulfill({ json: item.review })
  })
  await page.route(/\/api\/ui\/jobs\/[^/]+\/reviews\/\d+\/history(?:\?.*)?$/, route => {
    const url = new URL(route.request().url())
    const index = Number(url.pathname.split('/').at(-2))
    expect(url.searchParams.get('result_sha256')).toBe(resultHash)
    return route.fulfill({ json: { job_id: positive.job_id, result_sha256: resultHash, violation_index: index, entries: history.filter(record => record.violation_index === index), next_after_revision: 1, has_more: false } })
  })
  await restoreSyntheticJob(page)
  await page.goto('/')
  await expect(page.getByText('Ожидает решения человека', { exact: true })).toBeVisible()
  const serverTime = page.getByText('Интервал сервера после приёма: 12,5 с.', { exact: false })
  await expect(page.locator('.result-diagnostics')).toHaveJSProperty('open', false)
  await page.getByText('Технические сведения', { exact: true }).click()
  await expect(serverTime).toBeVisible()
  await expect(page.locator('[data-e2e-state="restored"]')).toContainText('Замер недоступен после восстановления')
  await page.getByLabel('Имя проверяющего', { exact: true }).fill('Синтетический проверяющий')
  await page.getByLabel('Комментарий', { exact: false }).fill('Синтетическое подтверждение.')
  await page.getByRole('button', { name: 'Подтвердить признак', exact: true }).click()
  await expect(page.getByText('Решение сохранено. Версия 1.', { exact: true })).toBeVisible()
  await expect(page.getByText('Признак подтверждён человеком', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'История решений', exact: true }).click()
  await expect(page.locator('.review-history')).toContainText('Синтетическое подтверждение.')
  await page.getByRole('button', { name: /Второе отдельное синтетическое событие/ }).click()
  await expect(page.getByText('Ожидает решения человека', { exact: true })).toBeVisible()
  await expect(page.getByLabel('Комментарий', { exact: false })).toHaveValue('')
  await page.getByLabel('Комментарий', { exact: false }).fill('Синтетическое отклонение.')
  await page.getByRole('button', { name: 'Отклонить признак', exact: true }).click()
  await expect(page.getByText('Признак отклонён человеком', { exact: true })).toBeVisible()
  await expect(page.getByText('Рассмотрено: 2 из 2')).toBeVisible()
  await expect(serverTime).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Обнаружены возможные признаки нарушений' })).toBeVisible()
  await page.getByRole('button', { name: /Синтетическое событие для проверки интерфейса/ }).click()
  await expect(page.locator('.human-review-status')).toContainText('Признак подтверждён человеком')
  await expect(page.getByLabel('Комментарий', { exact: false })).toHaveValue('Синтетическое подтверждение.')
  await page.reload()
  await expect(page.locator('.human-review-status')).toContainText('Признак подтверждён человеком')
  await expect(page.getByText('Рассмотрено: 2 из 2')).toBeVisible()
  await expect(page.locator('.result-diagnostics')).toHaveJSProperty('open', false)
  await page.getByText('Технические сведения', { exact: true }).click()
  await expect(serverTime).toBeVisible()
  await expect(page.locator('[data-e2e-state="restored"]')).toContainText('Замер недоступен после восстановления')
  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('button', { name: 'JSON модели', exact: true }).click()
  const download = await downloadPromise
  expect(fs.readFileSync((await download.path())!, 'utf8')).toBe(canonicalJson)
  await page.locator('.human-review').screenshot({ path: screenshot('human-confirmed-desktop') })
})

test('human review revision conflicts refresh the saved state and preserve drafts for an explicit retry', async ({ page }) => {
  await routes(page)
  const ledger = pendingReviews()
  let attempts = 0
  await page.route('**/api/ui/jobs/*/reviews', route => route.fulfill({ json: ledger }))
  await page.route(/\/api\/ui\/jobs\/[^/]+\/reviews\/0$/, route => {
    attempts += 1
    const input = route.request().postDataJSON()
    if (attempts === 1) {
      ledger.violations[0].review = { ...ledger.violations[0].review, state: 'rejected', revision: 1, claimed_reviewer_name: 'Другой синтетический проверяющий', comment: 'Синтетическое ранее сохранённое решение.', reviewed_at: '2026-10-17T11:59:00Z' }
      return route.fulfill({ status: 409, json: { detail: 'review_conflict', current_revision: 1, current_result_sha256: resultHash } })
    }
    expect(input.expected_revision).toBe(1)
    expect(input.claimed_reviewer_name).toBe('Синтетический черновик')
    expect(input.comment).toBe('Комментарий не должен исчезнуть.')
    ledger.violations[0].review = { ...ledger.violations[0].review, state: 'confirmed', revision: 2, claimed_reviewer_name: input.claimed_reviewer_name, comment: input.comment, reviewed_at: '2026-10-17T12:01:00Z' }
    return route.fulfill({ json: ledger.violations[0].review })
  })
  await restoreSyntheticJob(page)
  await page.goto('/')
  await expect(page.getByText('Ожидает решения человека', { exact: true })).toBeVisible()
  await page.getByLabel('Имя проверяющего', { exact: true }).fill('Синтетический черновик')
  await page.getByLabel('Комментарий', { exact: false }).fill('Комментарий не должен исчезнуть.')
  await page.getByRole('button', { name: 'Подтвердить признак', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('Конфликт версий')
  await expect(page.getByText('Признак отклонён человеком', { exact: true })).toBeVisible()
  await expect(page.getByLabel('Имя проверяющего', { exact: true })).toHaveValue('Синтетический черновик')
  await expect(page.getByLabel('Комментарий', { exact: false })).toHaveValue('Комментарий не должен исчезнуть.')
  await page.locator('.human-review').screenshot({ path: screenshot('human-conflict-desktop') })
  await page.getByRole('button', { name: 'Подтвердить признак', exact: true }).click()
  await expect(page.getByText('Решение сохранено. Версия 2.', { exact: true })).toBeVisible()
  expect(attempts).toBe(2)
})

test('unavailable review ledger cannot fabricate a pending or saved decision', async ({ page }) => {
  await routes(page)
  await page.route('**/api/ui/jobs/*/reviews', route => route.fulfill({ status: 503, json: { detail: 'synthetic_unavailable' } }))
  await restoreSyntheticJob(page)
  await page.goto('/')
  await expect(page.getByRole('button', { name: 'Подтвердить признак', exact: true })).toBeDisabled()
  await expect(page.getByRole('alert')).toBeVisible()
  await expect(page.getByText('Ожидает решения человека', { exact: true })).toHaveCount(0)
  await expect(page.getByRole('heading', { name: 'Обнаружены возможные признаки нарушений' })).toBeVisible()
})

test('chronological display saves the immutable source event index and evidence binding', async ({ page }) => {
  await routes(page)
  const model = syntheticPositive()
  model.violations[0].timecode_sec = 7.2
  model.violations[1].timecode_sec = 2.5
  await page.route('**/api/v1/jobs/*', route => route.fulfill({ json: model }))
  let savedIndex: number | null = null
  await page.route(/\/api\/ui\/jobs\/[^/]+\/reviews\/\d+$/, route => {
    const input = route.request().postDataJSON()
    savedIndex = Number(new URL(route.request().url()).pathname.split('/').at(-1))
    expect(savedIndex).toBe(1)
    expect(input.evidence_sha256).toBe(evidenceHash(1))
    return route.fulfill({ json: { ...pendingReview(1), state: 'confirmed', revision: 1, claimed_reviewer_name: input.claimed_reviewer_name, comment: input.comment, reviewed_at: '2026-10-17T12:00:00Z' } })
  })
  await restoreSyntheticJob(page)
  await page.goto('/')
  await expect(page.getByText('Ожидает решения человека', { exact: true })).toBeVisible()
  await page.getByLabel('Имя проверяющего', { exact: true }).fill('Синтетическая привязка')
  await page.getByRole('button', { name: 'Подтвердить признак', exact: true }).click()
  await expect(page.getByText('Решение сохранено. Версия 1.', { exact: true })).toBeVisible()
  expect(savedIndex).toBe(1)
})
