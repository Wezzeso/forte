import { useRef, useState } from 'react'
import { ArrowLineDownIcon, CheckIcon, CopyIcon, FilePdfIcon, LinkIcon, LockKeyIcon, CircleNotchIcon, WarningCircleIcon } from '@phosphor-icons/react'
import { friendlyError, getHumanReviews, getJuryJson, getMetadata, validateJob } from './api'
import type { HumanReviews, JuryJob } from './api'
import Modal from './Modal'
import { problemLink, reportTime } from './problem-links'

export default function ResultActions({ job, apiKey, selectedSourceIndex }: { job: JuryJob; apiKey: string; selectedSourceIndex?: number }) {
  const [dialog, setDialog] = useState<'pdf' | 'share' | null>(null)
  const [scope, setScope] = useState<'event' | 'all'>(selectedSourceIndex === undefined ? 'all' : 'event')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [copied, setCopied] = useState(false)
  const linkInput = useRef<HTMLInputElement>(null)
  const generating = useRef(false)
  const event = selectedSourceIndex === undefined ? undefined : job.violations?.[selectedSourceIndex]
  const link = problemLink(job.job_id, selectedSourceIndex, event?.timecode_sec)

  function open(next: 'pdf' | 'share') {
    setError(''); setMessage(''); setCopied(false)
    setScope(event ? 'event' : 'all'); setDialog(next)
  }

  async function copy() {
    setCopied(false); setError('')
    try {
      await navigator.clipboard.writeText(link)
      setCopied(true)
    } catch {
      linkInput.current?.focus(); linkInput.current?.select()
      setError('Не удалось скопировать автоматически. Ссылка выделена: скопируйте её вручную.')
    }
  }

  async function download() {
    if (generating.current) return
    generating.current = true
    setBusy(true); setError(''); setMessage('Получаем актуальный результат и сохранённые решения…')
    try {
      const [blob, details, ledger, renderer] = await Promise.all([
        getJuryJson(job.job_id, apiKey),
        getMetadata(job.job_id, apiKey, AbortSignal.timeout(30_000)).catch(() => null),
        getHumanReviews(job.job_id, apiKey).catch(() => null),
        import('./pdf-report'),
      ])
      const snapshot = validateJob(JSON.parse(await blob.text()), job.job_id)
      if (snapshot.status !== 'done') throw new Error('Проверка ещё не завершена. PDF доступен после получения полного результата.')
      // A review belongs to exact result bytes, never just a job/event label.
      const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', await blob.arrayBuffer()))].map(value => value.toString(16).padStart(2, '0')).join('')
      const reviews: HumanReviews | null = ledger?.result_sha256 === hash ? ledger : null
      if (JSON.stringify(snapshot.violations?.[selectedSourceIndex ?? -1]) !== JSON.stringify(event)) {
        throw new Error('Данные выбранного события изменились. Обновите страницу перед созданием отчёта.')
      }
      setMessage('Собираем PDF с исходными кадрами…')
      const report = await renderer.createPdfReport({ job: snapshot, metadata: details, reviews, resultHash: hash, sourceIndex: scope === 'event' ? selectedSourceIndex : undefined })
      const url = URL.createObjectURL(report)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `forte-${job.job_id}${scope === 'event' && selectedSourceIndex !== undefined ? `-event-${selectedSourceIndex + 1}` : '-report'}.pdf`
      anchor.click()
      setTimeout(() => URL.revokeObjectURL(url), 60_000)
      setMessage('PDF готов. Файл передан браузеру для скачивания.')
    } catch (failure) { setError(friendlyError(failure)); setMessage('') }
    finally { generating.current = false; setBusy(false) }
  }

  return <>
    <section className="result-actions" aria-label="Отчёт и ссылка на результат">
      <div className="result-actions-copy"><span className="detail-label">Сохранить и передать</span><p>{event ? <>Выбрано событие на <span className="timecode">{reportTime(event.timecode_sec)}</span></> : 'Полный результат проверки'}</p></div>
      <div className="result-actions-buttons"><button className="button button-secondary" onClick={() => open('share')}><LinkIcon size={18} />Поделиться ссылкой</button><button className="button button-primary" onClick={() => open('pdf')}><FilePdfIcon size={19} />Скачать PDF</button></div>
    </section>

    {dialog === 'share' && <Modal title="Ссылка на результат" onClose={() => setDialog(null)}>
      <div className="report-dialog">
        <div className="share-context"><span className="detail-label">{event ? `Событие · ${reportTime(event.timecode_sec)}` : 'Результат проверки'}</span><h3>{event?.type ?? 'События не выявлены'}</h3><p>{event ? 'Ссылка откроет эту задачу, выбранное событие и нужный момент исходной записи.' : 'Ссылка откроет полный результат этой задачи.'}</p></div>
        <label className="detail-label" htmlFor="problem-link">Ссылка для передачи</label><input ref={linkInput} id="problem-link" className="share-link-input" readOnly value={link} onFocus={e => e.currentTarget.select()} />
        <p className="share-access"><LockKeyIcon size={17} />Получателю понадобится свой ключ доступа к сервису. Ключ и видео не включены в ссылку.</p>
        {error && <p className="report-error" role="alert">{error}</p>}
        <div className="report-dialog-actions"><button className="button button-primary" onClick={() => { void copy() }}>{copied ? <CheckIcon size={18} /> : <CopyIcon size={18} />}{copied ? 'Ссылка скопирована' : 'Копировать ссылку'}</button></div>
        <span className="visually-hidden" role="status">{copied ? 'Ссылка скопирована' : ''}</span>
      </div>
    </Modal>}

    {dialog === 'pdf' && <Modal title="Отчёт в PDF" onClose={() => { if (!busy) setDialog(null) }}>
      <div className="report-dialog" aria-busy={busy}>
        <p className="report-intro">Результат модели, исходные кадры, источники и сохранённые решения человека — в одном документе.</p>
        <fieldset className="report-scope" disabled={busy}><legend>Что включить</legend>
          {event && <label className={scope === 'event' ? 'selected' : ''}><input type="radio" name="report-scope" value="event" checked={scope === 'event'} onChange={() => setScope('event')} /><span><strong>Выбранное событие</strong><small>{reportTime(event.timecode_sec)} · {event.type}</small></span></label>}
          <label className={scope === 'all' ? 'selected' : ''}><input type="radio" name="report-scope" value="all" checked={scope === 'all'} onChange={() => setScope('all')} /><span><strong>Вся проверка</strong><small>Все события ({job.violations?.length ?? 0}), кадры и общий итог</small></span></label>
        </fieldset>
        <div className="report-includes"><FilePdfIcon size={32} /><div><strong>В оформлении FORTE Panopticon</strong><p>Кликабельные ссылки на события и источники. Текст доступен для поиска и копирования.</p></div></div>
        <p className="report-note">Включаются только сохранённые решения. Черновики комментариев остаются в форме. Если дополнительные данные недоступны, это будет отмечено в отчёте.</p>
        {message && <p className="report-progress" role="status">{busy ? <CircleNotchIcon size={17} className="spin" /> : <CheckIcon size={17} />}{message}</p>}
        {error && <p className="report-error" role="alert"><WarningCircleIcon size={18} />{error}</p>}
        <div className="report-dialog-actions"><button className="button button-primary" disabled={busy || !apiKey} onClick={() => { void download() }}>{busy ? <CircleNotchIcon size={18} className="spin" /> : <ArrowLineDownIcon size={18} />}{busy ? 'Создаём отчёт…' : 'Сформировать и скачать'}</button></div>
      </div>
    </Modal>}
  </>
}
