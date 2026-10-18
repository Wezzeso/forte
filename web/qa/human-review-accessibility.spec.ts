/** Synthetic browser checks only. All requests and decisions stay inside Playwright routes. */
import { test, expect } from '@playwright/test'
import type { Page } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'

const fixture = JSON.parse(fs.readFileSync(path.resolve('../contracts/fixtures/positive.json'), 'utf8'))
const model = { ...fixture, violations: [
  { ...fixture.violations[0], type: 'Первое синтетическое событие', timecode_sec: 2.5 },
  { ...fixture.violations[0], type: 'Второе синтетическое событие', timecode_sec: 7.2 },
] }
const key = 'synthetic-review-accessibility-only'
const resultHash = 'a'.repeat(64)
const evidenceHash = (index: number) => String(index + 1).repeat(64)
function pending(index: number) {
  return { job_id: model.job_id, result_sha256: resultHash, violation_index: index, evidence_sha256: evidenceHash(index), state: 'pending', revision: 0, claimed_reviewer_name: null as string | null, comment: '', reviewed_at: null as string | null, reviewer_identity_verified: false }
}
function ledger() {
  return { job_id: model.job_id, result_sha256: resultHash, source_sha256: 'b'.repeat(64), reviewer_identity_verified: false, violations: [0, 1].map(index => ({ violation_index: index, evidence_sha256: evidenceHash(index), review: pending(index) })) }
}
async function prepare(page: Page) {
  await page.route('**/runtime-config.json', route => route.fulfill({ status: 404, json: {} }))
  await page.route('**/readyz', route => route.fulfill({ json: { status: 'ready', components: { database: true, storage: true, model: true, worker: true } } }))
  await page.route('**/api/v1/jobs/*', route => route.fulfill({ json: model }))
  await page.route('**/api/ui/jobs/*', route => route.fulfill({ json: { job_id: model.job_id, stage: 'done', elapsed_sec: 12.5 } }))
  await page.route('**/api/ui/jobs/*/video', route => route.fulfill({ contentType: 'video/quicktime', path: path.resolve('qa/fixtures/review-synthetic.mov') }))
  await page.route('**/api/ui/jobs/*/reviews', route => route.fulfill({ json: ledger() }))
  await page.route(/\/api\/ui\/jobs\/[^/]+\/reviews\/\d+\/history(?:\?.*)?$/, route => {
    const url = new URL(route.request().url())
    const index = Number(url.pathname.split('/').at(-2))
    return route.fulfill({ json: { job_id: model.job_id, result_sha256: resultHash, violation_index: index, entries: [], next_after_revision: 0, has_more: false } })
  })
  await page.addInitScript(({ key, job }) => {
    sessionStorage.setItem('forte.session.key', key)
    sessionStorage.setItem('forte.session.job', job)
  }, { key, job: model.job_id })
}
async function openReview(page: Page) {
  await page.goto('/')
  await expect(page.locator('.human-review-status')).toContainText('Ожидает решения человека')
}
async function draft(page: Page) {
  await page.getByLabel('Имя проверяющего', { exact: true }).fill('Синтетический черновик')
  await page.getByLabel('Комментарий', { exact: false }).fill('Синтетический комментарий сохранён в форме.')
}

test('review name disclaimer, required validation, keyboard history and per-finding drafts are accessible', async ({ page }) => {
  await prepare(page)
  let writes = 0
  page.on('request', request => { if (request.method() === 'POST') writes += 1 })
  await openReview(page)
  const name = page.getByLabel('Имя проверяющего', { exact: true })
  const comment = page.getByLabel('Комментарий', { exact: false })
  const confirm = page.getByRole('button', { name: 'Подтвердить признак', exact: true })
  await expect(name).toHaveAccessibleDescription('Имя указано пользователем, без проверки личности.')
  await expect(name).toHaveAttribute('required', '')
  await expect(comment).toHaveAccessibleDescription('0 / 4000')
  await confirm.click()
  await expect(name).toBeFocused()
  await name.fill('   ')
  await confirm.click()
  await expect(name).toBeFocused()
  expect(writes).toBe(0)
  await draft(page)
  await name.focus()
  await page.keyboard.press('Tab')
  await expect(comment).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(confirm).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(page.getByRole('button', { name: 'Отклонить признак', exact: true })).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(page.getByRole('button', { name: 'Обновить решения', exact: true })).toBeFocused()
  await page.keyboard.press('Tab')
  const history = page.getByRole('button', { name: 'История решений', exact: true })
  await expect(history).toBeFocused()
  await expect(history).toHaveAttribute('aria-expanded', 'false')
  await page.keyboard.press('Enter')
  const region = page.getByRole('region', { name: 'История решений этого события' })
  await expect(region).toBeVisible()
  await expect(region).toContainText('Сохранённых решений пока нет.')
  const hide = page.getByRole('button', { name: 'Скрыть историю', exact: true })
  await expect(hide).toHaveAttribute('aria-expanded', 'true')
  expect(await hide.getAttribute('aria-controls')).toBe(await region.getAttribute('id'))
  await hide.focus()
  await page.keyboard.press('Space')
  await expect(page.getByRole('region', { name: 'История решений этого события' })).toHaveCount(0)
  await page.getByRole('button', { name: /Второе синтетическое событие/ }).click()
  await expect(comment).toHaveValue('')
  await comment.fill('Отдельный черновик второго события.')
  await page.getByRole('button', { name: /Первое синтетическое событие/ }).click()
  await expect(comment).toHaveValue('Синтетический комментарий сохранён в форме.')
  await expect(name).toHaveValue('Синтетический черновик')
  expect(writes).toBe(0)
})

test('lost conflict refresh blocks repeat writes, retains drafts, and reconnect reads the current revision', async ({ page }) => {
  await prepare(page)
  const saved = ledger()
  let disconnected = false
  let writes = 0
  await page.route('**/api/ui/jobs/*/reviews', route => disconnected ? route.abort('internetdisconnected') : route.fulfill({ json: saved }))
  await page.route(/\/api\/ui\/jobs\/[^/]+\/reviews\/0$/, route => {
    writes += 1
    const input = route.request().postDataJSON()
    if (writes === 1) {
      saved.violations[0].review = { ...pending(0), state: 'rejected', revision: 1, claimed_reviewer_name: 'Другой синтетический проверяющий', comment: 'Ранее сохранённый синтетический отказ.', reviewed_at: '2026-10-17T12:00:00Z' }
      disconnected = true
      return route.fulfill({ status: 409, json: { detail: 'synthetic_revision_conflict' } })
    }
    expect(input.expected_revision).toBe(1)
    expect(input.result_sha256).toBe(resultHash)
    expect(input.evidence_sha256).toBe(evidenceHash(0))
    saved.violations[0].review = { ...saved.violations[0].review, state: 'confirmed', revision: 2, claimed_reviewer_name: input.claimed_reviewer_name, comment: input.comment, reviewed_at: '2026-10-17T12:01:00Z' }
    return route.fulfill({ json: saved.violations[0].review })
  })
  await openReview(page)
  await draft(page)
  const confirm = page.getByRole('button', { name: 'Подтвердить признак', exact: true })
  await confirm.click()
  await expect(page.locator('.human-review')).toContainText('Нужно обновить сохранённые решения.')
  await expect(confirm).toBeDisabled()
  await expect(page.locator('.human-review')).toContainText('Актуальность не подтверждена.')
  expect(writes).toBe(1)
  disconnected = false
  await page.getByRole('button', { name: 'Обновить решения', exact: true }).click()
  await expect(page.locator('.human-review-status')).toContainText('Признак отклонён человеком')
  await expect(page.locator('.saved-review')).toContainText('Указанное имя: Другой синтетический проверяющий')
  await expect(page.getByLabel('Имя проверяющего', { exact: true })).toHaveValue('Синтетический черновик')
  await expect(page.getByLabel('Комментарий', { exact: false })).toHaveValue('Синтетический комментарий сохранён в форме.')
  await expect(confirm).toBeEnabled()
  await confirm.click()
  await expect(page.getByText('Решение сохранено. Версия 2.', { exact: true })).toBeVisible()
  expect(writes).toBe(2)
})

test('an uncertain disconnected save requires a ledger refresh before another explicit decision', async ({ page }) => {
  await prepare(page)
  let writes = 0
  await page.route(/\/api\/ui\/jobs\/[^/]+\/reviews\/0$/, route => { writes += 1; return route.abort('internetdisconnected') })
  await openReview(page)
  await draft(page)
  const confirm = page.getByRole('button', { name: 'Подтвердить признак', exact: true })
  await confirm.click()
  await expect(page.locator('.review-error')).toContainText('Нажмите «Обновить решения»')
  await expect(confirm).toBeDisabled()
  expect(writes).toBe(1)
  await page.getByRole('button', { name: 'Обновить решения', exact: true }).click()
  await expect(confirm).toBeEnabled()
  await expect(page.locator('.human-review-status')).toContainText('Ожидает решения человека')
  await expect(page.getByLabel('Комментарий', { exact: false })).toHaveValue('Синтетический комментарий сохранён в форме.')
  expect(writes).toBe(1)
})

for (const changed of ['result', 'evidence'] as const) {
  test(`a refreshed ${changed} hash cannot silently rebind a decision to unseen content`, async ({ page }) => {
    await prepare(page)
    await openReview(page)
    await draft(page)
    const changedLedger = ledger()
    if (changed === 'result') {
      changedLedger.result_sha256 = 'c'.repeat(64)
      changedLedger.violations.forEach(item => { item.review.result_sha256 = changedLedger.result_sha256 })
    } else {
      changedLedger.violations[0].evidence_sha256 = 'd'.repeat(64)
      changedLedger.violations[0].review.evidence_sha256 = 'd'.repeat(64)
    }
    await page.route('**/api/ui/jobs/*/reviews', route => route.fulfill({ json: changedLedger }))
    await page.getByRole('button', { name: 'Обновить решения', exact: true }).click()
    await expect(page.locator('.human-review [role=alert]')).toContainText('Привязка результата или кадров изменилась.')
    await expect(page.getByRole('button', { name: 'Подтвердить признак', exact: true })).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Отклонить признак', exact: true })).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Обновить решения', exact: true })).toBeDisabled()
    await expect(page.locator('.human-review-status')).toHaveCount(0)
    await expect(page.getByLabel('Комментарий', { exact: false })).toHaveValue('Синтетический комментарий сохранён в форме.')
  })
}

test('history with a mismatched evidence binding remains undisplayed', async ({ page }) => {
  await prepare(page)
  await page.route(/\/api\/ui\/jobs\/[^/]+\/reviews\/0\/history(?:\?.*)?$/, route => route.fulfill({ json: {
    job_id: model.job_id, result_sha256: resultHash, violation_index: 0,
    entries: [{ ...pending(0), evidence_sha256: 'd'.repeat(64), state: 'confirmed', revision: 1, claimed_reviewer_name: 'Синтетический чужой кадр', reviewed_at: '2026-10-17T12:00:00Z' }],
    next_after_revision: 1, has_more: false,
  } }))
  await openReview(page)
  await page.getByRole('button', { name: 'История решений', exact: true }).click()
  await expect(page.locator('.review-history [role=alert]')).toContainText('История относится к другим кадрам.')
  await expect(page.locator('.review-history-entry')).toHaveCount(0)
  await expect(page.locator('.human-review-status')).toContainText('Ожидает решения человека')
})
