/** Synthetic PDF and deep-link checks. No private footage or production keys. */
import { test, expect } from '@playwright/test'
import type { Page } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'

const fixture = JSON.parse(fs.readFileSync(path.resolve('../contracts/fixtures/positive.json'), 'utf8'))
const key = 'synthetic-report-only-key'
const jobId = 'synthetic-pdf-sharing'
const model = { ...fixture, job_id: jobId, violations: [
  { ...fixture.violations[0], timecode_sec: 7.2, type: 'Позднее синтетическое событие', description: 'Синтетическое описание позднего события для проверки отчёта.' },
  { ...fixture.violations[0], timecode_sec: 2.5, type: 'Раннее синтетическое событие', description: 'Синтетическое описание раннего события для проверки отчёта.' },
] }
const resultBytes = JSON.stringify(model)
const resultHash = createHash('sha256').update(resultBytes).digest('hex')
function ledger() {
  return { job_id: jobId, result_sha256: resultHash, source_sha256: 'b'.repeat(64), reviewer_identity_verified: false, violations: [0, 1].map(index => ({
    violation_index: index, evidence_sha256: String(index + 1).repeat(64),
    review: { job_id: jobId, result_sha256: resultHash, evidence_sha256: String(index + 1).repeat(64), violation_index: index, state: index === 0 ? 'confirmed' : 'pending', revision: index === 0 ? 1 : 0, claimed_reviewer_name: index === 0 ? 'Синтетический специалист' : null, comment: index === 0 ? 'Сохранённый комментарий для отчёта.' : '', reviewed_at: index === 0 ? '2026-10-18T06:00:00Z' : null, reviewer_identity_verified: false },
  })) }
}
async function prepare(page: Page, authenticated = true) {
  await page.route('**/runtime-config.json', route => route.fulfill({ status: 404, json: {} }))
  await page.route('**/readyz', route => route.fulfill({ json: { status: 'ready' } }))
  await page.route('**/api/ui/config', route => route.fulfill({ json: { max_upload_bytes: 1024 ** 3, accepted_suffixes: ['.mov'] } }))
  await page.route('**/api/v1/jobs/*', route => {
    expect(route.request().headers()['x-api-key']).toBe(key)
    expect(route.request().url()).toContain(jobId)
    return route.fulfill({ contentType: 'application/json', body: resultBytes })
  })
  await page.route('**/api/ui/jobs/*', route => route.fulfill({ json: { job_id: jobId, stage: 'done', completed_at: '2026-10-18T05:59:00Z', limitations: ['Синтетическое ограничение анализа.'], events: [
    { observed_facts: ['Поздний синтетический факт.'], rule: { title: 'Синтетический источник позднего события', clause: 'TEST-LATE', url: 'https://example.com/late' } },
    { observed_facts: ['Ранний синтетический факт.'], rule: { title: 'Синтетический источник раннего события', clause: 'TEST-EARLY', url: 'https://example.com/early' } },
  ] } }))
  await page.route('**/api/ui/jobs/*/video', route => route.fulfill({ contentType: 'video/quicktime', path: path.resolve('qa/fixtures/review-synthetic.mov') }))
  await page.route('**/api/ui/jobs/*/reviews', route => route.fulfill({ json: ledger() }))
  if (authenticated) await page.addInitScript(key => {
    sessionStorage.setItem('forte.session.key', key)
    sessionStorage.setItem('forte.session.job', 'unrelated-stored-job')
  }, key)
}
async function savePdf(page: Page, name: string) {
  const downloading = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Сформировать и скачать', exact: true }).click()
  const download = await downloading
  const file = path.resolve(`qa/reports/${name}.pdf`)
  await download.saveAs(file)
  expect(fs.readFileSync(file).subarray(0, 5).toString()).toBe('%PDF-')
  const text = execFileSync('pdftotext', ['-enc', 'UTF-8', file, '-'], { encoding: 'utf8' })
  return { file, text, filename: download.suggestedFilename() }
}

test('links override stored jobs, preserve canonical event identity, and seek after video load', async ({ page }) => {
  await prepare(page)
  await page.goto(`/?job=${jobId}&event=0&t=7.2`)
  await expect(page.locator('.event-card.active')).toContainText('Позднее синтетическое событие')
  await expect.poll(() => page.locator('video').evaluate((video: HTMLVideoElement) => video.currentTime)).toBeCloseTo(7.2, 1)
  await page.getByRole('button', { name: 'Поделиться ссылкой' }).click()
  const link = await page.getByLabel('Ссылка для передачи').inputValue()
  expect(new URL(link).searchParams.get('event')).toBe('0')
  expect(link).not.toContain(key)
  await page.keyboard.press('Escape')
  await page.getByRole('button', { name: /Раннее синтетическое событие/ }).click()
  await page.reload()
  await expect(page.locator('.event-card.active')).toContainText('Раннее синтетическое событие')
  await expect.poll(() => page.locator('video').evaluate((video: HTMLVideoElement) => video.currentTime)).toBeCloseTo(2.5, 1)
})

test('a recipient connects their own key before any authenticated task request', async ({ page }) => {
  await prepare(page, false)
  let reads = 0
  page.on('request', request => { if (request.url().includes('/api/v1/jobs/')) reads++ })
  await page.goto(`/?job=${jobId}&event=0&t=7.2`)
  await expect(page.getByRole('dialog', { name: 'Подключить доступ' })).toBeVisible()
  expect(reads).toBe(0)
  await page.getByLabel('Ключ доступа', { exact: true }).fill(key)
  await page.getByRole('button', { name: 'Подключить', exact: true }).click()
  await expect(page.locator('.event-card.active')).toContainText('Позднее синтетическое событие')
  await expect.poll(() => page.locator('video').evaluate((video: HTMLVideoElement) => video.currentTime)).toBeCloseTo(7.2, 1)
})

test('copy succeeds and clipboard denial leaves a selectable manual fallback', async ({ page, context }) => {
  await prepare(page)
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  await page.goto(`/?job=${jobId}&event=0&t=7.2&secret=never-copy-this`)
  await page.getByRole('button', { name: 'Поделиться ссылкой' }).click()
  await page.getByRole('button', { name: 'Копировать ссылку' }).click()
  await expect(page.getByRole('button', { name: 'Ссылка скопирована' })).toBeVisible()
  const copied = await page.evaluate(() => navigator.clipboard.readText())
  expect(copied).toBe(`http://127.0.0.1:5173/?job=${jobId}&event=0&t=7.2`)
  await page.evaluate(() => { Object.defineProperty(navigator.clipboard, 'writeText', { configurable: true, value: () => Promise.reject(new Error('denied')) }) })
  await page.getByRole('button', { name: 'Ссылка скопирована' }).click()
  await expect(page.getByRole('alert')).toContainText('скопируйте её вручную')
  await expect(page.getByLabel('Ссылка для передачи')).toBeFocused()
})

test('selected-event PDF has Cyrillic text, evidence, bound saved decisions and exact clickable links', async ({ page }) => {
  await prepare(page)
  const errors: string[] = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto(`/?job=${jobId}&event=0&t=7.2`)
  await expect(page.locator('.event-card.active')).toContainText('Позднее синтетическое событие')
  await page.getByLabel('Комментарий', { exact: false }).fill('НЕСОХРАНЁННЫЙ ЧЕРНОВИК')
  await page.screenshot({ path: 'qa/screenshots/report-workspace-desktop.png', fullPage: true })
  await page.getByRole('button', { name: 'Скачать PDF', exact: true }).click()
  await page.screenshot({ path: 'qa/screenshots/report-dialog-desktop.png' })
  const pdf = await savePdf(page, 'selected-event')
  expect(pdf.filename).toBe(`forte-${jobId}-event-1.pdf`)
  expect(pdf.text).toContain('Позднее синтетическое событие')
  expect(pdf.text).not.toContain('Раннее синтетическое событие')
  expect(pdf.text).toContain('Признак подтверждён человеком')
  expect(pdf.text).toContain('Сохранённый комментарий для отчёта.')
  expect(pdf.text).not.toContain('НЕСОХРАНЁННЫЙ ЧЕРНОВИК')
  expect(pdf.text).toContain(resultHash)
  expect(pdf.text).toContain('Кадр 1')
  const raw = fs.readFileSync(pdf.file).toString('latin1')
  expect(raw).toContain(`/URI (http://127.0.0.1:5173/?job=${jobId}&event=0&t=7.2)`)
  expect(raw).toContain('/URI (https://example.com/late)')
  expect(raw).not.toContain(key)
  expect(errors).toEqual([])
})

test('full report includes all events in time order and survives long saved comments', async ({ page }) => {
  await prepare(page)
  const saved = ledger()
  saved.violations[0].review.comment = 'Синтетический длинный комментарий. '.repeat(105) + 'КОНЕЦ КОММЕНТАРИЯ'
  await page.route('**/api/ui/jobs/*/reviews', route => route.fulfill({ json: saved }))
  await page.goto(`/?job=${jobId}`)
  await page.getByRole('button', { name: 'Скачать PDF', exact: true }).click()
  await page.getByRole('radio', { name: /Вся проверка/ }).check()
  const pdf = await savePdf(page, 'full-report-long-comment')
  expect(pdf.text).toContain('Раннее синтетическое событие')
  expect(pdf.text).toContain('Позднее синтетическое событие')
  expect(pdf.text.indexOf('Раннее синтетическое событие')).toBeLessThan(pdf.text.indexOf('Позднее синтетическое событие'))
  expect(pdf.text).toContain('КОНЕЦ КОММЕНТАРИЯ')
  expect(pdf.text).toContain('TEST-EARLY')
  expect(pdf.text).toContain('TEST-LATE')
})

test('unavailable optional data is stated explicitly and a failed authoritative fetch can retry', async ({ page }) => {
  await prepare(page)
  await page.goto(`/?job=${jobId}`)
  await expect(page.locator('.event-card.active')).toBeVisible()
  await page.route('**/api/ui/jobs/*', route => route.fulfill({ status: 503, json: {} }))
  await page.route('**/api/ui/jobs/*/reviews', route => route.fulfill({ status: 503, json: {} }))
  await page.getByRole('button', { name: 'Скачать PDF', exact: true }).click()
  await page.route('**/api/v1/jobs/*', route => route.fulfill({ status: 503, json: {} }))
  await page.getByRole('button', { name: 'Сформировать и скачать' }).click()
  await expect(page.getByRole('dialog').getByRole('alert')).toBeVisible()
  await page.unroute('**/api/v1/jobs/*')
  await page.route('**/api/v1/jobs/*', route => route.fulfill({ contentType: 'application/json', body: resultBytes }))
  const pdf = await savePdf(page, 'optional-data-unavailable')
  expect(pdf.text).toContain('Дополнительные данные недоступны')
  expect(pdf.text).toContain('Сохранённые решения недоступны')
  expect(pdf.text).not.toContain('Признак подтверждён человеком')
})

test('reviews bound to different result bytes are never included in the PDF', async ({ page }) => {
  await prepare(page)
  const other = ledger()
  other.result_sha256 = 'a'.repeat(64)
  other.violations.forEach(item => { item.review.result_sha256 = other.result_sha256 })
  await page.route('**/api/ui/jobs/*/reviews', route => route.fulfill({ json: other }))
  await page.goto(`/?job=${jobId}`)
  await page.getByRole('button', { name: 'Скачать PDF', exact: true }).click()
  const pdf = await savePdf(page, 'mismatched-review-binding')
  expect(pdf.text).toContain('Сохранённые решения недоступны')
  expect(pdf.text).not.toContain('Синтетический специалист')
})

test('mobile dialogs fit the viewport and unknown shared events show a useful recovery', async ({ page }) => {
  await prepare(page)
  await page.setViewportSize({ width: 360, height: 800 })
  await page.goto(`/?job=${jobId}&event=999&t=7.2`)
  await expect(page.getByRole('alert')).toContainText('Событие из ссылки не найдено')
  await page.getByRole('button', { name: 'Поделиться ссылкой' }).click()
  await page.screenshot({ path: 'qa/screenshots/report-share-mobile.png' })
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(360)
  await page.keyboard.press('Escape')
  await expect(page.getByRole('button', { name: 'Поделиться ссылкой' })).toBeFocused()
  await page.getByRole('button', { name: 'Скачать PDF', exact: true }).click()
  await page.screenshot({ path: 'qa/screenshots/report-dialog-mobile.png' })
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(360)
})

test('no-finding checks can be shared and exported without inventing events or review states', async ({ page }) => {
  await prepare(page)
  const clear = { job_id: jobId, status: 'done', verdict: 'no_violation', violations: [] }
  await page.route('**/api/v1/jobs/*', route => route.fulfill({ json: clear }))
  await page.goto(`/?job=${jobId}`)
  await page.getByRole('button', { name: 'Поделиться ссылкой' }).click()
  expect(await page.getByLabel('Ссылка для передачи').inputValue()).toBe(`http://127.0.0.1:5173/?job=${jobId}`)
  await page.keyboard.press('Escape')
  await page.getByRole('button', { name: 'Скачать PDF', exact: true }).click()
  await expect(page.getByRole('radio')).toHaveCount(1)
  const pdf = await savePdf(page, 'no-findings')
  expect(pdf.text).toContain('Признаки нарушений не обнаружены')
  expect(pdf.text).toContain('Событий: 0')
  expect(pdf.text).not.toContain('Признак подтверждён человеком')
})

test('malformed task links never silently restore an unrelated saved job', async ({ page }) => {
  await prepare(page)
  let reads = 0
  page.on('request', request => { if (request.url().includes('/api/v1/jobs/')) reads++ })
  await page.goto('/?job=..%2Finvalid')
  await expect(page.getByRole('alert')).toContainText('Ссылка на задачу некорректна')
  expect(reads).toBe(0)
  await expect(page.getByRole('button', { name: 'Скачать PDF' })).toHaveCount(0)
})
