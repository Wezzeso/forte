import { apiUrl } from './runtime-config'

export interface EvidenceFrame {
  timestamp_sec: number
  image_base64: string
}

export interface Violation {
  type: string
  timecode_sec: number
  description: string
  evidence_frames: EvidenceFrame[]
}

export interface JuryJob {
  job_id: string
  status: 'processing' | 'done' | 'failed'
  verdict?: 'violation' | 'no_violation'
  violations?: Violation[]
}

export interface RuleReference {
  title: string
  clause?: string
  url?: string
  observation?: string
  limitations?: string[]
}

export interface JobMetadata {
  job_id: string
  stage?: string
  created_at?: string
  deadline_at?: string
  updated_at?: string
  completed_at?: string
  elapsed_sec?: number
  error_code?: string
  limitations?: string[]
  rules?: RuleReference[]
  events?: { rule?: RuleReference; limitations?: string[]; observed_facts?: string[] }[]
}

export interface Readiness {
  status: 'ready' | 'not_ready'
  components?: { database: boolean; storage: boolean; model: boolean }
}

export interface UiConfig {
  max_upload_bytes: number
  accepted_suffixes: string[]
}

export interface HumanReviewRecord {
  job_id: string
  result_sha256: string
  violation_index: number
  evidence_sha256: string
  state: 'pending' | 'confirmed' | 'rejected'
  revision: number
  claimed_reviewer_name: string | null
  comment: string
  reviewed_at: string | null
  reviewer_identity_verified: false
}

export interface HumanReviews {
  job_id: string
  result_sha256: string
  source_sha256: string
  reviewer_identity_verified: false
  violations: { violation_index: number; evidence_sha256: string; review: HumanReviewRecord }[]
}

export interface HumanReviewHistory {
  job_id: string
  result_sha256: string
  violation_index: number
  entries: HumanReviewRecord[]
  next_after_revision: number
  has_more: boolean
}

export interface HumanReviewInput {
  result_sha256: string
  evidence_sha256: string
  decision: 'confirm' | 'reject'
  claimed_reviewer_name: string
  comment: string
  expected_revision: number
}

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message) }
}

async function request(path: string, key: string, options: RequestInit = {}): Promise<Response> {
  const signal = options.signal ?? AbortSignal.timeout(30_000)
  const response = await fetch(apiUrl(path), {
    ...options,
    signal,
    headers: { ...options.headers, 'X-API-Key': key },
    cache: 'no-store',
    credentials: 'omit',
    redirect: 'error',
  })
  if (!response.ok) {
    const messages: Record<number, string> = {
      401: 'Ключ доступа не принят. Проверьте его и повторите запрос.',
      403: 'Доступ к задаче запрещён. Проверьте ключ доступа.',
      404: 'Задача не найдена на сервере.',
      413: 'Видео превышает допустимый размер загрузки на сервере.',
      415: 'Сервер не поддерживает этот формат видео. Выберите файл MOV.',
      422: 'Сервер не смог принять видео. Проверьте формат файла.',
      429: 'Сервис занят. Подождите немного и повторите запрос.',
      409: 'Решение или данные события изменились. Обновите сохранённое решение перед повторной отправкой.',
      503: 'Сервис пока не готов принимать видео. Повторите позже.',
    }
    throw new ApiError(response.status, messages[response.status] ?? `Сервер вернул ошибку (${response.status}). Повторите запрос.`)
  }
  return response
}

export async function uploadVideo(file: File, key: string, signal: AbortSignal): Promise<string> {
  const body = new FormData()
  body.append('file', file)
  const response = await request('/api/v1/analyze', key, { method: 'POST', body, signal })
  const data: unknown = await response.json()
  if (!data || typeof data !== 'object' || !('job_id' in data) || typeof data.job_id !== 'string' || !data.job_id) {
    throw new Error('Сервис принял запрос, но не вернул номер задачи. Проверьте подключение перед повторной загрузкой.')
  }
  return data.job_id
}

export async function getUiConfig(key: string, signal: AbortSignal): Promise<UiConfig> {
  const raw: unknown = await (await request('/api/ui/config', key, { signal })).json()
  if (!raw || typeof raw !== 'object') throw new Error('Ограничения загрузки не получены.')
  const config = raw as UiConfig
  if (!Number.isSafeInteger(config.max_upload_bytes) || config.max_upload_bytes <= 0 || !Array.isArray(config.accepted_suffixes) || !config.accepted_suffixes.every(value => typeof value === 'string') || !config.accepted_suffixes.includes('.mov')) {
    throw new Error('Сервис не подтвердил ограничения загрузки MOV.')
  }
  return config
}

export async function getJob(id: string, key: string, signal: AbortSignal): Promise<JuryJob> {
  const response = await request(`/api/v1/jobs/${encodeURIComponent(id)}`, key, { signal })
  const raw: unknown = await response.json()
  return validateJob(raw, id)
}

export function validateJob(raw: unknown, id: string): JuryJob {
  if (!raw || typeof raw !== 'object') throw new Error('Ответ сервиса не соответствует формату задачи.')
  const data = raw as JuryJob
  if (data.job_id !== id || !['processing', 'done', 'failed'].includes(data.status)) {
    throw new Error('Ответ сервиса не соответствует формату задачи.')
  }
  if (data.status === 'done') {
    if (!['violation', 'no_violation'].includes(data.verdict ?? '') || !Array.isArray(data.violations)) {
      throw new Error('В ответе сервиса отсутствует полный результат проверки.')
    }
    if ((data.verdict === 'violation') !== (data.violations.length > 0)) {
      throw new Error('Вердикт сервиса не согласован со списком событий.')
    }
    for (const event of data.violations) {
      if (typeof event.type !== 'string' || typeof event.description !== 'string' || !Number.isFinite(event.timecode_sec) || event.timecode_sec < 0 || !Array.isArray(event.evidence_frames) || event.evidence_frames.length < 1 || event.evidence_frames.length > 3 || event.evidence_frames.some(frame => !Number.isFinite(frame.timestamp_sec) || frame.timestamp_sec < 0 || typeof frame.image_base64 !== 'string' || !frame.image_base64.startsWith('/9j/'))) {
        throw new Error('В результате есть событие без корректного времени или JPEG-кадров. Проверка результата не завершена.')
      }
    }
  }
  return data
}

export async function getMetadata(id: string, key: string, signal: AbortSignal): Promise<JobMetadata | null> {
  const response = await request(`/api/ui/jobs/${encodeURIComponent(id)}`, key, { signal })
  const data = await response.json() as JobMetadata
  if (data.job_id !== id) return null
  return data
}

export async function getVideo(id: string, key: string, signal: AbortSignal): Promise<Blob> {
  return (await request(`/api/ui/jobs/${encodeURIComponent(id)}/video`, key, { signal })).blob()
}

export async function getJuryJson(id: string, key: string): Promise<Blob> {
  return (await request(`/api/v1/jobs/${encodeURIComponent(id)}`, key)).blob()
}

function validReview(record: HumanReviewRecord, id: string, resultHash: string, index: number): boolean {
  return !!record && record.job_id === id && record.result_sha256 === resultHash && record.violation_index === index
    && ['pending', 'confirmed', 'rejected'].includes(record.state) && Number.isInteger(record.revision) && record.revision >= 0
    && typeof record.evidence_sha256 === 'string' && /^[a-f0-9]{64}$/i.test(record.evidence_sha256)
    && typeof record.comment === 'string' && (record.claimed_reviewer_name === null || typeof record.claimed_reviewer_name === 'string')
}

export async function getHumanReviews(id: string, key: string, signal?: AbortSignal): Promise<HumanReviews> {
  const data = await (await request(`/api/ui/jobs/${encodeURIComponent(id)}/reviews`, key, { signal })).json() as HumanReviews
  if (!data || data.job_id !== id || typeof data.result_sha256 !== 'string' || !/^[a-f0-9]{64}$/i.test(data.result_sha256) || !Array.isArray(data.violations)
    || data.violations.some(item => !Number.isInteger(item.violation_index) || !validReview(item.review, id, data.result_sha256, item.violation_index) || item.evidence_sha256 !== item.review.evidence_sha256)) {
    throw new Error('Сервис вернул некорректную привязку решений к результату. Сохранение заблокировано.')
  }
  return data
}

export async function saveHumanReview(id: string, index: number, key: string, input: HumanReviewInput): Promise<HumanReviewRecord> {
  const data = await (await request(`/api/ui/jobs/${encodeURIComponent(id)}/reviews/${index}`, key, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(input),
  })).json() as HumanReviewRecord
  if (!validReview(data, id, input.result_sha256, index) || data.evidence_sha256 !== input.evidence_sha256 || data.revision !== input.expected_revision + 1 || data.state !== (input.decision === 'confirm' ? 'confirmed' : 'rejected')) {
    throw new Error('Не удалось проверить подтверждение сохранённого решения. Обновите данные перед повторной отправкой.')
  }
  return data
}

export async function getHumanReviewHistory(id: string, index: number, resultHash: string, key: string, afterRevision = 0): Promise<HumanReviewHistory> {
  const query = new URLSearchParams({ result_sha256: resultHash, after_revision: String(afterRevision), limit: '50' })
  const data = await (await request(`/api/ui/jobs/${encodeURIComponent(id)}/reviews/${index}/history?${query}`, key)).json() as HumanReviewHistory
  if (!data || data.job_id !== id || data.result_sha256 !== resultHash || data.violation_index !== index || !Array.isArray(data.entries)
    || data.entries.some(record => !validReview(record, id, resultHash, index)) || !Number.isInteger(data.next_after_revision) || typeof data.has_more !== 'boolean') {
    throw new Error('Не удалось проверить историю решений для этого события.')
  }
  return data
}

export function friendlyError(error: unknown): string {
  if (error instanceof ApiError) return error.message
  if (error instanceof DOMException && error.name === 'TimeoutError') return 'Сервис не ответил вовремя. Проверьте подключение и повторите запрос.'
  if (error instanceof TypeError) return 'Нет соединения с сервисом. Проверяем подключение; задача сохраняется на сервере.'
  return error instanceof Error ? error.message : 'Не удалось выполнить запрос. Повторите попытку.'
}

export function safeSourceUrl(value?: string): string | undefined {
  if (!value) return undefined
  try { const parsed = new URL(value); return ['https:', 'http:'].includes(parsed.protocol) ? parsed.href : undefined } catch { return undefined }
}

