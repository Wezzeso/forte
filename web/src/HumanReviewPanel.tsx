import { useEffect, useId, useRef, useState } from 'react'
import {
  CheckIcon as Check, WarningCircleIcon as CircleAlert, ClockIcon as Clock3,
  ClockCounterClockwiseIcon as History, CircleNotchIcon as LoaderCircle,
  ArrowsClockwiseIcon as RefreshCw, UserIcon as UserRound, XIcon as X,
} from '@phosphor-icons/react'
import { ApiError, friendlyError, getHumanReviewHistory, getHumanReviews, saveHumanReview } from './api'
import type { HumanReviewHistory, HumanReviews } from './api'

function reviewTime(value: string | null): string {
  if (!value || !Number.isFinite(Date.parse(value))) return 'Время не указано'
  return new Date(value).toLocaleString('ru-RU', { timeZone: 'Asia/Qyzylorda', day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' })
}

const LABELS = { pending: 'Ожидает решения человека', confirmed: 'Признак подтверждён человеком', rejected: 'Признак отклонён человеком' }

export default function HumanReviewPanel({ jobId, apiKey, violationIndex }: { jobId: string; apiKey: string; violationIndex: number }) {
  // A different job or access session must never inherit another result's drafts or ledger.
  return <BoundHumanReviewPanel key={`${jobId}:${apiKey}`} jobId={jobId} apiKey={apiKey} violationIndex={violationIndex} />
}

function sameBindings(previous: HumanReviews, next: HumanReviews): boolean {
  return previous.result_sha256 === next.result_sha256 && previous.source_sha256 === next.source_sha256
    && previous.violations.length === next.violations.length
    && previous.violations.every(item => next.violations.find(candidate => candidate.violation_index === item.violation_index)?.evidence_sha256 === item.evidence_sha256)
}

function BoundHumanReviewPanel({ jobId, apiKey, violationIndex }: { jobId: string; apiKey: string; violationIndex: number }) {
  const id = useId()
  const [reviews, setReviews] = useState<HumanReviews | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState('')
  const [saving, setSaving] = useState(false)
  const [saveMessage, setSaveMessage] = useState<Record<number, string>>({})
  const [saveError, setSaveError] = useState<Record<number, string>>({})
  const [reviewerName, setReviewerName] = useState('')
  const [comments, setComments] = useState<Record<number, string>>({})
  const [histories, setHistories] = useState<Record<number, HumanReviewHistory>>({})
  const [historyOpen, setHistoryOpen] = useState<Record<number, boolean>>({})
  const [historyError, setHistoryError] = useState<Record<number, string>>({})
  const [historyLoading, setHistoryLoading] = useState<Record<number, boolean>>({})
  const [needsRefresh, setNeedsRefresh] = useState(false)
  const [bindingChanged, setBindingChanged] = useState(false)
  const [reload, setReload] = useState(0)
  const form = useRef<HTMLFormElement>(null)
  const initialBindings = useRef<HumanReviews | null>(null)
  const requestGeneration = useRef(0)
  const binding = reviews?.violations.find(item => item.violation_index === violationIndex)
  const review = binding?.review
  const reviewedCount = reviews?.violations.filter(item => item.review.state !== 'pending').length ?? 0

  function acceptReviews(data: HumanReviews): boolean {
    if (initialBindings.current && !sameBindings(initialBindings.current, data)) {
      setBindingChanged(true)
      setReviews(null)
      setLoadError('Привязка результата или кадров изменилась. Перезагрузите страницу и заново просмотрите результат перед сохранением. Черновик оставлен в форме.')
      return false
    }
    initialBindings.current ??= data
    setReviews(data)
    setNeedsRefresh(false)
    setSaveError({})
    setSaveMessage({})
    return true
  }

  useEffect(() => {
    const controller = new AbortController()
    requestGeneration.current += 1
    setSaving(false)
    setHistoryLoading({})
    setHistories({})
    setHistoryOpen({})
    setHistoryError({})
    setLoading(true)
    setLoadError('')
    getHumanReviews(jobId, apiKey, controller.signal).then(data => { if (!controller.signal.aborted) acceptReviews(data) }).catch(error => {
      if (!controller.signal.aborted) setLoadError(friendlyError(error))
    }).finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => { controller.abort(); requestGeneration.current += 1 }
  }, [jobId, apiKey, reload])

  async function save(decision: 'confirm' | 'reject') {
    if (!reviews || !binding || loading || loadError || bindingChanged || needsRefresh || saving || !form.current?.reportValidity()) return
    const index = violationIndex
    const generation = requestGeneration.current
    setSaving(true)
    setSaveError(previous => ({ ...previous, [index]: '' }))
    setSaveMessage(previous => ({ ...previous, [index]: '' }))
    try {
      const record = await saveHumanReview(jobId, index, apiKey, {
        result_sha256: reviews.result_sha256, evidence_sha256: binding.evidence_sha256,
        decision, claimed_reviewer_name: reviewerName.trim(), comment: comments[index] ?? '', expected_revision: binding.review.revision,
      })
      if (generation !== requestGeneration.current) return
      setReviews(previous => previous ? { ...previous, violations: previous.violations.map(item => item.violation_index === index ? { ...item, review: record } : item) } : previous)
      setSaveMessage(previous => ({ ...previous, [index]: `Решение сохранено. Версия ${record.revision}.` }))
      setHistories(previous => { const next = { ...previous }; delete next[index]; return next })
      setHistoryOpen(previous => ({ ...previous, [index]: false }))
    } catch (error) {
      if (generation !== requestGeneration.current) return
      if (error instanceof ApiError && error.status === 409) {
        try {
          const latest = await getHumanReviews(jobId, apiKey)
          if (generation !== requestGeneration.current) return
          if (acceptReviews(latest)) {
            setHistories({})
            setHistoryOpen({})
            setSaveError(previous => ({ ...previous, [index]: 'Конфликт версий: показано актуальное сохранённое решение. Ваши имя и комментарий сохранены в форме. Проверьте изменения перед повторной отправкой.' }))
          }
        } catch {
          if (generation !== requestGeneration.current) return
          setNeedsRefresh(true)
          setSaveError(previous => ({ ...previous, [index]: 'Решение изменилось, но обновить его не удалось. Черновик сохранён. Нажмите «Обновить решения» перед повторной отправкой.' }))
          setLoadError('Нужно обновить сохранённые решения.')
        }
      } else {
        // The server may have committed a disconnected request. Read its revision before any retry.
        setNeedsRefresh(true)
        setSaveError(previous => ({ ...previous, [index]: `${friendlyError(error)} Черновик сохранён в форме. Нажмите «Обновить решения» перед повторной отправкой.` }))
      }
    } finally { if (generation === requestGeneration.current) setSaving(false) }
  }

  async function loadHistory(more = false) {
    if (!reviews || !binding || loading || loadError || bindingChanged || historyLoading[violationIndex]) return
    const index = violationIndex
    const generation = requestGeneration.current
    setHistoryOpen(previous => ({ ...previous, [index]: true }))
    setHistoryLoading(previous => ({ ...previous, [index]: true }))
    setHistoryError(previous => ({ ...previous, [index]: '' }))
    try {
      const previous = more ? histories[index] : undefined
      const data = await getHumanReviewHistory(jobId, index, reviews.result_sha256, apiKey, previous?.next_after_revision ?? 0)
      if (generation !== requestGeneration.current) return
      if (data.entries.some(record => record.evidence_sha256 !== binding.evidence_sha256)) {
        throw new Error('История относится к другим кадрам. Обновите решения перед просмотром истории.')
      }
      setHistories(histories => ({ ...histories, [index]: { ...data, entries: [...(previous?.entries ?? []), ...data.entries] } }))
    } catch (error) {
      if (generation === requestGeneration.current) setHistoryError(previous => ({ ...previous, [index]: friendlyError(error) }))
    } finally { if (generation === requestGeneration.current) setHistoryLoading(previous => ({ ...previous, [index]: false })) }
  }

  const blocked = loading || !!loadError || !binding || bindingChanged || needsRefresh || saving

  return <section className="human-review" aria-labelledby={`${id}-title`} aria-busy={loading || saving}>
    <div className="human-review-heading"><UserRound size={19} aria-hidden="true" /><h3 id={`${id}-title`}>Решение человека</h3></div>
    <p className="human-review-explainer" id={`${id}-explainer`}>Вывод модели предварительный. Подтвердите или отклоните признак после просмотра записи и основания.</p>
    {loading && <p className="review-loading" role="status"><LoaderCircle size={15} className="spin" />Загружаем сохранённые решения…</p>}
    {loadError && <div className="review-error" role="alert"><CircleAlert size={17} /><p>{loadError}</p></div>}
    {!loading && !loadError && !binding && <div className="review-error" role="alert"><CircleAlert size={17} /><p>Привязка этого события к результату не получена. Сохранение недоступно.</p></div>}
    {review && <>
      {loadError && <p className="review-separation-note">Показано последнее полученное решение. Актуальность не подтверждена.</p>}
      <div className={`human-review-status review-${review.state}`} role="status">{review.state === 'pending' ? <Clock3 size={16} /> : review.state === 'confirmed' ? <Check size={16} /> : <X size={16} />}<strong>{LABELS[review.state]}</strong></div>
      <div className="review-summary"><span>Рассмотрено: {reviewedCount} из {reviews?.violations.length}</span><span>Версия {review.revision}</span></div>
      {review.state !== 'pending' && <div className="saved-review"><span>Указанное имя: {review.claimed_reviewer_name} · {reviewTime(review.reviewed_at)} (UTC+5)</span>{review.comment && <p>{review.comment}</p>}</div>}
    </>}
    <form className="human-review-form" ref={form} aria-describedby={`${id}-explainer`} onSubmit={event => event.preventDefault()}>
      <label htmlFor={`${id}-name`}>Имя проверяющего</label>
      <input id={`${id}-name`} name="claimed_reviewer_name" value={reviewerName} onChange={event => setReviewerName(event.target.value)} maxLength={120} pattern={'.*\\S.*'} title="Укажите имя, а не только пробелы" aria-describedby={`${id}-name-hint`} required disabled={saving} autoComplete="off" placeholder="Укажите имя" />
      <small id={`${id}-name-hint`}>Имя указано пользователем, без проверки личности.</small>
      <label htmlFor={`${id}-comment`}>Комментарий <span>необязательно</span></label>
      <textarea id={`${id}-comment`} name="comment" value={comments[violationIndex] ?? ''} onChange={event => setComments(previous => ({ ...previous, [violationIndex]: event.target.value }))} maxLength={4000} aria-describedby={`${id}-comment-count`} disabled={saving} rows={3} placeholder="Что подтвердило или опровергло признак" />
      <span className="comment-count" id={`${id}-comment-count`}>{(comments[violationIndex] ?? '').length} / 4000</span>
      <div className="review-decision-actions"><button type="button" className="button review-confirm" disabled={blocked} onClick={() => { void save('confirm') }}><Check size={16} aria-hidden="true" />Подтвердить признак</button><button type="button" className="button review-reject" disabled={blocked} onClick={() => { void save('reject') }}><X size={16} aria-hidden="true" />Отклонить признак</button></div>
    </form>
    {saving && <p className="review-loading" role="status"><LoaderCircle size={15} className="spin" />Сохраняем решение…</p>}
    {saveMessage[violationIndex] && <p className="review-save-success" role="status">{saveMessage[violationIndex]}</p>}
    {saveError[violationIndex] && <div className="review-error" role="alert"><CircleAlert size={17} /><p>{saveError[violationIndex]}</p></div>}
    <div className="review-tools"><button className="text-link" type="button" disabled={loading || saving || bindingChanged} onClick={() => setReload(value => value + 1)}><RefreshCw size={14} aria-hidden="true" />Обновить решения</button><button className="text-link" type="button" aria-expanded={!!historyOpen[violationIndex]} aria-controls={`${id}-history`} disabled={loading || !!loadError || !binding || bindingChanged || !!historyLoading[violationIndex] || saving} onClick={() => { if (historyOpen[violationIndex]) setHistoryOpen(previous => ({ ...previous, [violationIndex]: false })); else void loadHistory() }}><History size={14} aria-hidden="true" />{historyOpen[violationIndex] ? 'Скрыть историю' : 'История решений'}</button></div>
    <div className="review-history" id={`${id}-history`} role="region" aria-label="История решений этого события" hidden={!historyOpen[violationIndex]} aria-busy={!!historyLoading[violationIndex]}>
      {historyLoading[violationIndex] && <p role="status">Загружаем историю…</p>}
      {historyError[violationIndex] && <p className="review-error" role="alert">{historyError[violationIndex]}</p>}
      {histories[violationIndex]?.entries.map(record => <article className="review-history-entry" key={record.revision}><div><strong>{LABELS[record.state]}</strong><span>Версия {record.revision}</span></div><p>Указанное имя: {record.claimed_reviewer_name} · {reviewTime(record.reviewed_at)} (UTC+5)</p>{record.comment && <blockquote>{record.comment}</blockquote>}</article>)}
      {histories[violationIndex]?.entries.length === 0 && <p>Сохранённых решений пока нет.</p>}
      {histories[violationIndex]?.has_more && <button className="button button-secondary button-small" type="button" disabled={!!historyLoading[violationIndex]} onClick={() => { void loadHistory(true) }}>Загрузить ещё</button>}
    </div>
    <p className="review-separation-note">Решения хранятся отдельно. Исходный результат модели и время анализа не изменяются.</p>
  </section>
}
