import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { CSSProperties, FormEvent } from 'react'
import {
  ArrowLineDownIcon as ArrowDownToLine, ArrowLeftIcon as ArrowLeft,
  ArrowRightIcon as ArrowRight, ArrowUpRightIcon as ArrowUpRight, CaretRightIcon as ChevronRight,
  WarningCircleIcon as CircleAlert, ClockIcon as Clock3, ArrowSquareOutIcon as ExternalLink,
  FileVideoIcon as FileVideo2, KeyIcon as KeyRound, ListIcon as Menu,
  CircleNotchIcon as LoaderCircle, LockKeyIcon as LockKeyhole, CornersOutIcon as Maximize2,
  PlayIcon as Play, PlusIcon as Plus, ArrowsClockwiseIcon as RefreshCw,
  ShieldCheckIcon as ShieldCheck, UserIcon as UserRound, VideoCameraIcon as Video, XIcon as X,
} from '@phosphor-icons/react'
import { ApiError, friendlyError, getJob, getJuryJson, getMetadata, getUiConfig, getVideo, safeSourceUrl, uploadVideo } from './api'
import type { EvidenceFrame, JobMetadata, JuryJob, Readiness } from './api'
import { apiUrl } from './runtime-config'
import HumanReviewPanel from './HumanReviewPanel'
import Illustration from './Illustration'
import Modal from './Modal'
import ResultActions from './ResultActions'
import { problemLink, readProblemTarget } from './problem-links'
import useSmoothScrolling from './useSmoothScrolling'

const KEY_STORAGE = 'forte.session.key'
const JOB_STORAGE = 'forte.session.job'

function stored(name: string): string {
  try { return sessionStorage.getItem(name) ?? '' } catch { return '' }
}
function remember(name: string, value: string): void {
  try { if (value) sessionStorage.setItem(name, value); else sessionStorage.removeItem(name) } catch { /* Session may disallow storage. */ }
}
function formatTime(seconds: number): string {
  const safe = Math.max(0, seconds)
  const whole = Math.floor(safe)
  const minutes = Math.floor(whole / 60)
  return `${String(minutes).padStart(2, '0')}:${String(whole % 60).padStart(2, '0')}`
}
function preciseTime(seconds: number): string {
  const milliseconds = Math.round(Math.max(0, Number.isFinite(seconds) ? seconds : 0) * 1000)
  return `${formatTime(Math.floor(milliseconds / 1000))}.${String(milliseconds % 1000).padStart(3, '0')}`
}
function elapsedTime(seconds: number): string { return `${Math.max(0, seconds).toLocaleString('ru-RU', { minimumFractionDigits: 1, maximumFractionDigits: 1 })} с` }
function fileSize(bytes: number): string { return `${(bytes / 1024 / 1024).toLocaleString('ru-RU', { maximumFractionDigits: 1 })} МБ` }
function eventCount(count: number): string {
  const last = count % 10
  return `${count} ${count % 100 >= 11 && count % 100 <= 14 ? 'событий' : last === 1 ? 'событие' : last >= 2 && last <= 4 ? 'события' : 'событий'}`
}

const STAGES: Record<string, string> = {
  queued: 'Видео ожидает обработки', claimed: 'Задача принята в работу',
  decoding: 'Подготавливаем кадры', decode: 'Подготавливаем кадры',
  screening: 'Анализируем видеоряд', analyzing: 'Анализируем видеоряд',
  refining: 'Уточняем наблюдения', assembling: 'Собираем доказательства',
  evidence: 'Собираем доказательства', publishing: 'Сохраняем результат',
  processing: 'Проверяем видео', done: 'Проверка завершена', failed: 'Обработка не завершена',
}

function EvidenceImage({ frame, alt, className }: { frame: EvidenceFrame; alt: string; className?: string }) {
  const [broken, setBroken] = useState(false)
  useEffect(() => setBroken(false), [frame.image_base64])
  if (broken) return <span className="broken-evidence" role="status"><CircleAlert size={22} /><span>Не удалось открыть JPEG-кадр</span></span>
  return <img className={className} src={`data:image/jpeg;base64,${frame.image_base64}`} alt={alt} onError={() => setBroken(true)} />
}

export default function App() {
  const resetPageScroll = useSmoothScrolling()
  const [menuOpen, setMenuOpen] = useState(false)
  const [key, setKey] = useState(() => stored(KEY_STORAGE))
  const [keyDraft, setKeyDraft] = useState('')
  const [keyOpen, setKeyOpen] = useState(() => !!readProblemTarget() && !stored(KEY_STORAGE))
  const [file, setFile] = useState<File | null>(null)
  const [localVideoUrl, setLocalVideoUrl] = useState('')
  const [remoteVideoUrl, setRemoteVideoUrl] = useState('')
  const [videoError, setVideoError] = useState('')
  const [downloadingResult, setDownloadingResult] = useState(false)
  const [downloadError, setDownloadError] = useState('')
  const [jobId, setJobId] = useState(() => readProblemTarget()?.jobId ?? (new URLSearchParams(location.search).has('job') ? '' : stored(JOB_STORAGE)))
  const [job, setJob] = useState<JuryJob | null>(null)
  const [metadata, setMetadata] = useState<JobMetadata | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadUnknown, setUploadUnknown] = useState(false)
  const [error, setError] = useState('')
  const [pollError, setPollError] = useState('')
  const [missingJob, setMissingJob] = useState(false)
  const [retry, setRetry] = useState(0)
  const [dragging, setDragging] = useState(false)
  const [readiness, setReadiness] = useState<Readiness | null>(null)
  const [maxUploadBytes, setMaxUploadBytes] = useState<number | null>(null)
  const [configLoading, setConfigLoading] = useState(false)
  const [selected, setSelected] = useState(0)
  const [enlargedFrame, setEnlargedFrame] = useState<EvidenceFrame | null>(null)
  const [now, setNow] = useState(Date.now())
  const [monotonicNow, setMonotonicNow] = useState(() => performance.now())
  const [clientDuration, setClientDuration] = useState<number | null>(null)
  const [failedClientDuration, setFailedClientDuration] = useState<number | null>(null)
  const [displayReadyAt, setDisplayReadyAt] = useState<number | null>(null)
  const [evidenceReady, setEvidenceReady] = useState(false)
  const [evidenceError, setEvidenceError] = useState('')
  const [readySourceUrl, setReadySourceUrl] = useState('')
  const [duration, setDuration] = useState(0)
  const [videoAspect, setVideoAspect] = useState(9 / 16)
  const [currentTime, setCurrentTime] = useState(0)
  const fileInput = useRef<HTMLInputElement>(null)
  const player = useRef<HTMLVideoElement>(null)
  const uploadAbort = useRef<AbortController | null>(null)
  const resultHeading = useRef<HTMLHeadingElement>(null)
  const processingHeading = useRef<HTMLHeadingElement>(null)
  const uploadStartedAt = useRef<number | null>(null)
  const menuToggle = useRef<HTMLButtonElement>(null)
  const linkTarget = useRef(readProblemTarget())
  const pendingSeek = useRef<number | null>(null)
  const [linkError, setLinkError] = useState(() => new URLSearchParams(location.search).has('job') && !readProblemTarget() ? 'Ссылка на задачу некорректна. Проверьте её или загрузите новое видео.' : '')

  useEffect(() => {
    if (!menuOpen) return
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    document.querySelector<HTMLElement>('#mobile-navigation a')?.focus()
    const desktop = window.matchMedia('(min-width: 1024px)')
    const closeOnDesktop = () => { if (desktop.matches) setMenuOpen(false) }
    const onMenuKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        setMenuOpen(false)
        menuToggle.current?.focus()
      }
      if (event.key === 'Tab') {
        const controls = [menuToggle.current, ...document.querySelectorAll<HTMLElement>('#mobile-navigation a, #mobile-navigation button')]
          .filter((control): control is HTMLElement => !!control && control.getClientRects().length > 0)
        const first = controls[0]
        const last = controls.at(-1)
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
        else if (!controls.includes(document.activeElement as HTMLElement)) { event.preventDefault(); first?.focus() }
      }
    }
    closeOnDesktop()
    desktop.addEventListener('change', closeOnDesktop)
    document.addEventListener('keydown', onMenuKey)
    return () => {
      document.body.style.overflow = previousOverflow
      desktop.removeEventListener('change', closeOnDesktop)
      document.removeEventListener('keydown', onMenuKey)
    }
  }, [menuOpen])

  const busy = uploading || !!jobId && (!job || job.status === 'processing')
  const fileTooLarge = !!file && maxUploadBytes !== null && file.size > maxUploadBytes
  const servicePreparing = readiness?.status === 'not_ready'
  const completed = job?.status === 'done'
  const failed = job?.status === 'failed'
  const hasResult = completed || failed
  const screen = hasResult ? 'result' : busy ? 'processing' : 'upload'
  const events = (job?.violations ?? []).map((event, sourceIndex) => ({ ...event, sourceIndex })).sort((a, b) => a.timecode_sec - b.timecode_sec || a.sourceIndex - b.sourceIndex)
  const activeEvent = events[selected]
  const sourceUrl = localVideoUrl || remoteVideoUrl
  const portraitVideo = videoAspect < 1
  const sourceReady = !!sourceUrl && readySourceUrl === sourceUrl
  const metadataStart = metadata?.created_at ? Date.parse(metadata.created_at) : NaN
  const metadataEnd = metadata?.completed_at ? Date.parse(metadata.completed_at) : NaN
  const serverElapsed = typeof metadata?.elapsed_sec === 'number' && Number.isFinite(metadata.elapsed_sec) ? Math.max(0, metadata.elapsed_sec) : Number.isFinite(metadataStart) && (!hasResult || Number.isFinite(metadataEnd)) ? Math.max(0, ((Number.isFinite(metadataEnd) ? metadataEnd : now) - metadataStart) / 1000) : null
  const elapsed = clientDuration ?? (uploadStartedAt.current === null ? null : Math.max(0, (monotonicNow - uploadStartedAt.current) / 1000))
  const humanElapsed = displayReadyAt === null ? null : Math.max(0, (monotonicNow - displayReadyAt) / 1000)
  const limits = Array.isArray(metadata?.limitations) ? metadata.limitations.filter(v => typeof v === 'string') : []
  const activeMeta = metadata?.events?.[activeEvent?.sourceIndex ?? selected]
  const activeRule = activeMeta?.rule ?? metadata?.rules?.[activeEvent?.sourceIndex ?? selected]
  const ruleUrl = safeSourceUrl(activeRule?.url)

  useEffect(() => {
    const restore = () => {
      const target = readProblemTarget()
      linkTarget.current = target
      pendingSeek.current = null
      setSelected(0)
      setLinkError('')
      setFile(null); setJob(null); setMetadata(null); setReadySourceUrl('')
      setJobId(target?.jobId ?? '')
      remember(JOB_STORAGE, target?.jobId ?? '')
      setRetry(value => value + 1)
    }
    window.addEventListener('popstate', restore)
    return () => window.removeEventListener('popstate', restore)
  }, [])

  useEffect(() => {
    const target = linkTarget.current
    if (!completed || !target || target.jobId !== jobId) return
    const index = target.eventIndex === null ? 0 : events.findIndex(event => event.sourceIndex === target.eventIndex)
    if (index < 0) {
      setLinkError('Событие из ссылки не найдено в этой задаче. Показан полный результат проверки.')
    } else {
      setSelected(index)
      pendingSeek.current = target.seconds ?? events[index]?.timecode_sec ?? null
      if (player.current && player.current.readyState >= 1 && pendingSeek.current !== null) {
        player.current.currentTime = Math.min(pendingSeek.current, player.current.duration || pendingSeek.current)
        pendingSeek.current = null
      }
    }
    linkTarget.current = null
    remember(JOB_STORAGE, jobId)
  }, [completed, jobId, job])

  useEffect(() => {
    document.title = failed ? 'Проверка не завершена — FORTE Panopticon'
      : completed ? 'Результат проверки — FORTE Panopticon'
      : busy ? 'Проверяем видео — FORTE Panopticon'
      : 'FORTE Panopticon — Видеоаналитика'
  }, [busy, completed, failed])

  function navigateFromMenu(section: string) {
    setMenuOpen(false)
    requestAnimationFrame(() => document.getElementById(section)?.focus({ preventScroll: true }))
  }

  useEffect(() => {
    if (!file) { setLocalVideoUrl(''); return }
    const url = URL.createObjectURL(file)
    setLocalVideoUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => {
    let active = true
    const controller = new AbortController()
    async function check() {
      try {
        const response = await fetch(apiUrl('/readyz'), { cache: 'no-store', credentials: 'omit', redirect: 'error', signal: AbortSignal.any([controller.signal, AbortSignal.timeout(5000)]) })
        const data = await response.json() as Readiness
        if (active) setReadiness(['ready', 'not_ready'].includes(data.status) ? data : null)
      } catch { if (active) setReadiness(null) }
    }
    void check()
    const timer = setInterval(() => { void check() }, 30_000)
    return () => { active = false; controller.abort(); clearInterval(timer) }
  }, [])

  useEffect(() => {
    setMaxUploadBytes(null)
    setConfigLoading(false)
    if (!key) return
    const controller = new AbortController()
    const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(5000)])
    setConfigLoading(true)
    getUiConfig(key, signal).then(config => {
      if (!controller.signal.aborted) setMaxUploadBytes(config.max_upload_bytes)
    }).catch(() => {
      // Optional UI metadata must not invent a limit or bypass server validation.
    }).finally(() => { if (!controller.signal.aborted) setConfigLoading(false) })
    return () => controller.abort()
  }, [key, retry])

  useEffect(() => {
    if (!busy && !completed) return
    const timer = setInterval(() => { setNow(Date.now()); setMonotonicNow(performance.now()) }, 250)
    return () => clearInterval(timer)
  }, [busy, completed])

  useLayoutEffect(() => {
    if (screen === 'upload' || document.querySelector('[aria-modal="true"]')) return
    // Reset before paint when replacing the long upload page, cancelling old inertia.
    // Polling and receipt of the job ID stay in the same screen and preserve scrolling.
    resetPageScroll()
    const heading = screen === 'processing' ? processingHeading : resultHeading
    heading.current?.focus({ preventScroll: true })
  }, [screen, resetPageScroll])

  // Decode every result JPEG, including findings not currently selected. A done
  // response alone is not the end of an upload-to-useful-result measurement.
  useEffect(() => {
    setEvidenceReady(false)
    setEvidenceError('')
    if (job?.status !== 'done') return
    let active = true
    const images = (job.violations ?? []).flatMap(event => event.evidence_frames).map(frame => {
      const img = new Image()
      img.src = `data:image/jpeg;base64,${frame.image_base64}`
      return img
    })
    Promise.all(images.map(img => img.decode())).then(() => {
      if (active) setEvidenceReady(true)
    }).catch(() => {
      if (active) setEvidenceError('Не все исходные JPEG-кадры удалось открыть. Полное время до готового результата не зафиксировано.')
    })
    return () => { active = false }
  }, [job])

  useEffect(() => {
    if (!completed || !evidenceReady || !sourceReady || displayReadyAt !== null) return
    let secondFrame = 0
    const firstFrame = requestAnimationFrame(() => {
      secondFrame = requestAnimationFrame(() => {
        const readyAt = performance.now()
        setMonotonicNow(readyAt)
        setDisplayReadyAt(readyAt)
        if (uploadStartedAt.current !== null) setClientDuration(Math.max(0, (readyAt - uploadStartedAt.current) / 1000))
      })
    })
    return () => { cancelAnimationFrame(firstFrame); cancelAnimationFrame(secondFrame) }
  }, [completed, evidenceReady, sourceReady, displayReadyAt])

  useEffect(() => {
    if (failed && uploadStartedAt.current !== null) setFailedClientDuration(Math.max(0, (performance.now() - uploadStartedAt.current) / 1000))
  }, [failed])

  useEffect(() => {
    if (!jobId || !key) return
    let active = true
    let timer: ReturnType<typeof setTimeout> | undefined
    let failures = 0
    const pollingStartedAt = Date.now()
    const controller = new AbortController()
    async function poll() {
      const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(20_000)])
      const [result, details] = await Promise.allSettled([getJob(jobId, key, signal), getMetadata(jobId, key, signal)])
      if (!active) return
      if (details.status === 'fulfilled' && details.value) setMetadata(details.value)
      if (result.status === 'fulfilled') {
        failures = 0
        setJob(result.value)
        setPollError('')
        setMissingJob(false)
        if (result.value.status !== 'processing') {
          return
        }
      } else {
        if (controller.signal.aborted) return
        setPollError(friendlyError(result.reason))
        setMissingJob(result.reason instanceof ApiError && result.reason.status === 404)
        failures += 1
        if (result.reason instanceof ApiError && [401, 403, 404].includes(result.reason.status)) return
      }
      const interval = failures === 0 && Date.now() - pollingStartedAt < 10_000 ? 500 : 1500
      timer = setTimeout(() => { void poll() }, Math.min(interval * 2 ** failures, 12_000))
    }
    void poll()
    return () => { active = false; controller.abort(); if (timer) clearTimeout(timer) }
  }, [jobId, key, retry])

  useEffect(() => {
    if (!jobId || !key || file || !completed) return
    const controller = new AbortController()
    let url = ''
    setVideoError('')
    getVideo(jobId, key, controller.signal).then(blob => {
      if (controller.signal.aborted) return
      url = URL.createObjectURL(blob)
      setRemoteVideoUrl(url)
    }).catch(error => { if (!controller.signal.aborted) setVideoError(friendlyError(error)) })
    return () => { controller.abort(); if (url) URL.revokeObjectURL(url); setRemoteVideoUrl('') }
  }, [jobId, key, file, completed, retry])

  useEffect(() => () => uploadAbort.current?.abort(), [])

  function chooseFile(next?: File) {
    if (!next || busy) return
    setError('')
    if (!next.name.toLocaleLowerCase().endsWith('.mov')) { setError('Выберите видео в формате .mov.'); return }
    if (next.size === 0) { setError('Файл пуст. Выберите другое видео.'); return }
    setFile(next)
    setUploadUnknown(false)
  }

  function saveKey(event: FormEvent) {
    event.preventDefault()
    const value = keyDraft.trim()
    if (!value) return
    setKey(value)
    remember(KEY_STORAGE, value)
    setKeyDraft('')
    setKeyOpen(false)
    setRetry(value => value + 1)
  }

  async function startAnalysis() {
    if (!file || busy || servicePreparing || fileTooLarge) return
    if (!key) { setKeyOpen(true); return }
    setError('')
    setUploadUnknown(false)
    setUploading(true)
    setClientDuration(null)
    setFailedClientDuration(null)
    setDisplayReadyAt(null)
    setEvidenceError('')
    setReadySourceUrl('')
    const controller = new AbortController()
    uploadAbort.current = controller
    try {
      // Deliberately immediately before transfer, after auth/file checks.
      uploadStartedAt.current = performance.now()
      setMonotonicNow(uploadStartedAt.current)
      const id = await uploadVideo(file, key, controller.signal)
      setJobId(id)
      remember(JOB_STORAGE, id)
      window.history.replaceState(null, '', problemLink(id))
      setJob({ job_id: id, status: 'processing' })
    } catch (error) {
      setError(friendlyError(error))
      setUploadUnknown(!(error instanceof ApiError))
      if (error instanceof ApiError && [401, 403].includes(error.status)) setKeyOpen(true)
    } finally { setMonotonicNow(performance.now()); setUploading(false); uploadAbort.current = null }
  }

  function reset() {
    window.history.replaceState(null, '', window.location.pathname)
    linkTarget.current = null; pendingSeek.current = null; setLinkError('')
    remember(JOB_STORAGE, '')
    setJobId(''); setJob(null); setMetadata(null); setFile(null); setSelected(0)
    setError(''); setPollError(''); setMissingJob(false); uploadStartedAt.current = null
    setClientDuration(null); setFailedClientDuration(null); setDisplayReadyAt(null); setReadySourceUrl('')
    setEvidenceReady(false); setEvidenceError('')
    setDuration(0); setVideoAspect(9 / 16); setCurrentTime(0); setVideoError(''); setUploadUnknown(false)
    setDownloadError('')
    if (fileInput.current) fileInput.current.value = ''
  }

  function seek(seconds: number) {
    if (player.current) {
      player.current.currentTime = seconds
      setCurrentTime(seconds)
      player.current.focus()
    }
  }

  function selectEvent(index: number) {
    setSelected(index); seek(events[index].timecode_sec)
    window.history.replaceState(null, '', problemLink(jobId, events[index].sourceIndex, events[index].timecode_sec))
  }

  function sourceLoaded(video: HTMLVideoElement) {
    if (video.videoWidth > 0 && video.videoHeight > 0 && Number.isFinite(video.duration) && video.duration > 0) {
      setVideoAspect(video.videoWidth / video.videoHeight)
      setReadySourceUrl(video.currentSrc)
      if (pendingSeek.current !== null) {
        video.currentTime = Math.min(pendingSeek.current, video.duration)
        pendingSeek.current = null
      }
      setVideoError('')
    } else {
      setReadySourceUrl('')
      setVideoError('Браузер не подтвердил доступность видеодорожки и её длительность. Откройте оригинал в локальном проигрывателе; кадры событий доступны ниже.')
    }
  }

  async function downloadResult() {
    if (!jobId || !key || downloadingResult) return
    setDownloadingResult(true)
    setDownloadError('')
    try {
      const blob = await getJuryJson(jobId, key)
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `${jobId}-model-result.json`
      anchor.click()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (error) { setDownloadError(friendlyError(error)) }
    finally { setDownloadingResult(false) }
  }

  return <div className={`app-shell redesign-shell ${!hasResult && !busy ? 'landing-state' : 'workspace-state'}`}>
    <a className="skip-link" href="#main">К содержимому</a>
    <div className="main-shell">
      <header className={`topbar ${menuOpen ? 'menu-open' : ''}`}>
        <a className="brand-block" href="#main" aria-label="FORTE Panopticon" onClick={() => setMenuOpen(false)}><img className="brand-logo" src="/brand/forte-panopticon.svg" alt="FORTE Panopticon" width="731" height="314" /></a>
        <nav className="desktop-nav" aria-label="Навигация"><a href="#main">Обзор</a><a href="#upload">Проверка видео</a><a href="#how">Как это работает</a></nav>
        <button className={`key-button ${key ? 'key-connected' : ''}`} onClick={() => setKeyOpen(true)}><span>{key ? 'Доступ подключён' : 'Ключ доступа'}</span></button>
        <button ref={menuToggle} className="menu-toggle" aria-label={menuOpen ? 'Закрыть меню' : 'Открыть меню'} aria-expanded={menuOpen} aria-controls="mobile-navigation" onClick={() => setMenuOpen(value => !value)}>{menuOpen ? <X size={27} /> : <Menu size={27} />}</button>
      </header>
      {menuOpen && <nav className="mobile-navigation" id="mobile-navigation" aria-label="Мобильная навигация"><a href="#main" onClick={() => navigateFromMenu('main')}>Обзор</a><a href="#upload" onClick={() => navigateFromMenu('upload')}>Проверка видео</a><a href="#how" onClick={() => navigateFromMenu('how')}>Как это работает</a><button className="button" onClick={() => { setMenuOpen(false); setKeyOpen(true) }}>{key ? 'Доступ подключён' : 'Ключ доступа'}<ArrowRight size={20} /></button><img src="/zimran/wave-bottom-right.svg" alt="" /></nav>}

      <main id="main" className={`main-content ${hasResult ? 'result-content' : ''}`} tabIndex={-1} inert={menuOpen}>
        <div className={`page-heading ${hasResult ? 'compact-heading' : busy ? 'processing-heading' : 'upload-hero'}`}>
          <div className="hero-copy"><h1 aria-label={!hasResult && !busy ? 'Видим суть. Решаете вы.' : undefined}>{hasResult ? 'Результат проверки' : busy ? 'Проверяем ваше видео' : <><span className="hero-title-line"><span>Видим</span><img className="hero-portrait hero-portrait-first" src="/team/hero-hoodie.webp" width="1254" height="1254" alt="" aria-hidden="true" fetchPriority="high" /><span>суть.</span></span><span className="hero-title-line"><span>Решаете</span><img className="hero-portrait hero-portrait-second" src="/team/hero-blazer.webp" width="1254" height="1254" alt="" aria-hidden="true" fetchPriority="high" /><span>вы.</span></span></>}</h1>{!hasResult && <p>{busy ? 'Анализируем запись и собираем доказательства.' : 'Видеоаналитика FORTE Panopticon.'}</p>}{!hasResult && !busy && <a className="hero-cta" href="#upload">Проверить видео<ArrowRight size={20} /></a>}</div>
          {!hasResult && !busy && <img className="hero-wave" src="/zimran/zigzag-purple.svg" alt="" aria-hidden="true" />}
          {hasResult && <div className="heading-actions">{completed && <button className="button button-secondary new-analysis" disabled={downloadingResult} onClick={() => { void downloadResult() }}><ArrowDownToLine size={17} />{downloadingResult ? 'Получаем JSON…' : 'JSON модели'}</button>}<button className="button button-secondary new-analysis" onClick={reset}><Plus size={18} />Новое видео</button></div>}
        </div>
        {linkError && <div className="notice notice-warning" role="alert"><CircleAlert size={18} /><p>{linkError}</p></div>}
        {downloadError && <div className="notice notice-warning" role="alert"><CircleAlert size={18} /><p>{downloadError}</p></div>}

        {!!pollError && <div className="notice notice-warning" role="alert"><CircleAlert size={20} /><div><strong>Не удалось обновить статус</strong><p>{pollError}</p></div><button className="button button-small button-secondary" onClick={() => setRetry(value => value + 1)}><RefreshCw size={15} />Повторить</button>{missingJob && <button className="button button-small button-secondary" onClick={reset}>К загрузке</button>}</div>}

        {jobId && !key && <div className="notice notice-warning" role="status"><KeyRound size={20} /><div><strong>Подключите доступ для восстановления задачи</strong><p>Номер задачи сохранён в этой вкладке. Для просмотра нужен ключ доступа.</p></div><button className="button button-small button-secondary" onClick={() => setKeyOpen(true)}>Ввести ключ</button></div>}

        {!busy && !hasResult && <><div className="section-ticker" aria-hidden="true"><div>{[0, 1, 2, 3].map(item => <span key={item}>От записи к решению<img src="/zimran/zigzag-small-white.svg" alt="" /></span>)}</div></div><div className="upload-grid" id="upload" tabIndex={-1}>
          <section className="upload-panel panel" aria-labelledby="upload-title">
            <div className="panel-heading"><div><h2 id="upload-title">Добавьте видео</h2></div><span className="file-format">.MOV</span></div>
            <label className="visually-hidden" htmlFor="video-file">Исходное видео MOV</label><input id="video-file" className="visually-hidden" ref={fileInput} type="file" accept=".mov,video/quicktime" onChange={event => chooseFile(event.target.files?.[0])} tabIndex={-1} />
            <button className={`dropzone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`} onClick={() => fileInput.current?.click()} onDragOver={event => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); chooseFile(event.dataTransfer.files[0]) }} aria-label={file ? `Заменить видео ${file.name}` : 'Выбрать или перетащить видео MOV'}>
              <span className="upload-art">{file && localVideoUrl ? <video className="upload-preview" src={localVideoUrl} preload="auto" muted playsInline aria-hidden="true" tabIndex={-1} /> : <Illustration kind="upload" />}</span>
              <strong>{file ? file.name : 'Перетащите видео сюда'}</strong>
              <span>{file ? `${fileSize(file.size)} · Готово к отправке` : <>или <span className="inline-link">выберите файл</span></>}</span><span className="dropzone-hint">Вертикальные и горизонтальные записи · без обрезки</span>
            </button>
            <details className="file-details"><summary>Требования к видео</summary><p className="file-constraints" data-max-upload-bytes={maxUploadBytes ?? undefined}>{maxUploadBytes !== null ? `Лимит сервера: ${fileSize(maxUploadBytes)}. ` : configLoading ? 'Получаем текущий лимит размера… ' : key ? 'Текущий лимит размера не подтверждён. ' : 'Подключите доступ, чтобы увидеть текущий лимит размера. '}Длительность и кодек проверит сервер при приёме. Для воспроизведения MOV браузеру нужен совместимый видеокодек.</p></details>
            {(error || fileTooLarge) && <div className="inline-error" id="upload-error" role="alert"><CircleAlert size={17} /><span>{error || `Размер видео превышает лимит сервера: ${fileSize(maxUploadBytes!)}. Выберите файл меньшего размера.`}{uploadUnknown && <small>Подтверждение приёма не получено. Задача могла сохраниться на сервере; уточните это у оператора перед повторной загрузкой.</small>}</span></div>}
            {servicePreparing && <p id="service-preparing" className="readiness-note" role="status"><LoaderCircle size={16} className="spin" />Сервис готовится. Проверка станет доступна после подготовки.</p>}
            <div className="upload-footer"><span className={`service-status ${readiness?.status === 'ready' ? 'ready' : ''}`}><i />{readiness?.status === 'ready' ? 'Сервис готов' : readiness?.status === 'not_ready' ? 'Сервис готовится' : 'Связь не подтверждена'}</span><button className="button button-primary" disabled={!file || servicePreparing || fileTooLarge} aria-describedby={[servicePreparing ? 'service-preparing' : '', fileTooLarge ? 'upload-error' : ''].filter(Boolean).join(' ') || undefined} onClick={() => { void startAnalysis() }}>Начать проверку<ArrowRight size={18} /></button></div>
          </section>

          <aside className="how-panel" id="how" aria-labelledby="how-title" tabIndex={-1}><h2 id="how-title">Видеть детали.<br />Понимать целое.</h2><div className="process-steps">
            <div className="process-step"><Illustration kind="upload" /><div><h3>Загрузите запись</h3></div></div>
            <div className="process-step"><Illustration kind="inspect" /><div><h3>Изучите события и кадры</h3></div></div>
            <div className="process-step"><Illustration kind="decide" /><div><h3>Примите решение</h3></div></div>
          </div><div className="privacy-note"><ShieldCheck size={18} /><p>Без звука и распознавания лиц</p></div></aside>
        </div></>}

        {busy && <section className="processing-panel panel" aria-labelledby="processing-title">
          <div className="processing-visual" aria-hidden="true"><Illustration kind="analyze" /></div>
          <span className="status-pill pill-processing"><LoaderCircle size={14} className="spin" />{uploading ? 'Загрузка' : 'В обработке'}</span>
          <h2 id="processing-title" ref={processingHeading} tabIndex={-1} aria-live="polite">{uploading ? 'Передаём оригинальное видео' : STAGES[metadata?.stage ?? ''] ?? 'Проверяем видео'}</h2>
          <p>{uploading ? 'Ожидаем подтверждения приёма. Этот способ передачи не сообщает точный процент загрузки.' : 'Задача сохранена на сервере. Можно восстановить её в этой вкладке после обновления страницы.'}</p>
          <div className="processing-details"><div><span>{uploadStartedAt.current === null ? 'На сервере после приёма' : 'С начала загрузки'}</span><strong>{elapsed !== null ? elapsedTime(elapsed) : serverElapsed !== null ? elapsedTime(serverElapsed) : '—'}</strong></div><div><span>Задача</span><strong className="job-id">{jobId || 'Ожидаем номер'}</strong></div></div>
          {uploadStartedAt.current === null && <p className="timing-note">После восстановления вкладки полное время от начала загрузки недоступно.</p>}
          {file && <div className="processing-file"><FileVideo2 size={18} /><span>{file.name}</span><small>{fileSize(file.size)}</small></div>}
          {metadata?.stage && <span className="actual-stage">Этап сервиса: {metadata.stage}</span>}
        </section>}

        {hasResult && <>
          <section className={`verdict-banner ${failed ? 'verdict-failed' : job.verdict === 'violation' ? 'verdict-attention' : 'verdict-clear'}`} aria-labelledby="verdict-title">
            <Illustration className="verdict-art" kind={failed || job.verdict === 'violation' ? 'alert' : 'clear'} />
            <div className="verdict-text"><h2 id="verdict-title" ref={resultHeading} tabIndex={-1}>{failed ? 'Не удалось завершить проверку' : job.verdict === 'violation' ? 'Обнаружены возможные признаки нарушений' : 'Признаки нарушений не обнаружены'}</h2><p>{failed ? 'Результат не сформирован. Попробуйте загрузить запись повторно или передайте номер задачи оператору.' : job.verdict === 'violation' ? `${eventCount(events.length)} для внимательной проверки. Окончательный вывод принимает специалист.` : 'В проанализированном видеоряде модель не выявила признаков. Учитывайте качество и полноту записи.'}</p></div>
          </section>

          {completed && <ResultActions key={`${jobId}:${key}`} job={job} apiKey={key} selectedSourceIndex={activeEvent?.sourceIndex} />}

          {evidenceError && <div className="notice notice-warning" role="alert"><CircleAlert size={18} /><p>{evidenceError}</p><button className="button button-small button-secondary" onClick={() => setRetry(value => value + 1)}><RefreshCw size={15} />Повторить получение результата</button></div>}

          {failed && <div className="failure-actions"><button className="button button-primary" onClick={reset}><RefreshCw size={17} />Выбрать видео повторно</button>{metadata?.error_code && <span>Код: <code>{metadata.error_code}</code></span>}</div>}

          {!failed && <div className={`review-grid ${portraitVideo ? 'portrait-review' : 'landscape-review'}`} style={{ '--video-aspect': videoAspect } as CSSProperties}>
            <section className="video-panel panel" aria-label="Исходное видео и доказательства">
              <div className="review-panel-heading"><h3><Video size={18} />Исходная запись</h3><span>{file ? file.name : 'Видео задачи'}</span></div>
              <div className="video-surface">
                {sourceUrl ? <video ref={player} src={sourceUrl} controls preload="auto" playsInline muted aria-label="Исходная видеозапись без анализа звука" onLoadedMetadata={event => { const video = event.currentTarget; if (Number.isFinite(video.duration)) setDuration(video.duration); if (video.videoWidth > 0 && video.videoHeight > 0) setVideoAspect(video.videoWidth / video.videoHeight) }} onLoadedData={event => sourceLoaded(event.currentTarget)} onTimeUpdate={event => setCurrentTime(event.currentTarget.currentTime)} onError={() => { setReadySourceUrl(''); setVideoError('Браузер не смог воспроизвести этот кодек MOV. Откройте оригинал в локальном проигрывателе; кадры событий доступны ниже.') }} /> : <div className="video-placeholder"><Illustration kind="upload" /><p>{videoError || 'Загружаем исходную запись…'}</p></div>}
              </div>
              {videoError && sourceUrl && <div className="video-error" role="status"><CircleAlert size={16} /><p>{videoError}</p><a className="text-link" href={sourceUrl} download={file?.name ?? `${jobId}.mov`}><ArrowDownToLine size={15} />Скачать оригинал</a></div>}
              {videoError && !sourceUrl && <div className="video-error"><button className="button button-secondary" onClick={() => setRetry(value => value + 1)}><RefreshCw size={16} />Повторить загрузку записи</button></div>}
              <div className="source-timeline"><div className="timeline-labels"><span>События на записи</span><span className="timecode">{formatTime(currentTime)} / {formatTime(duration)}</span></div><div className="timeline-track"><span className="timeline-progress" style={{ width: `${duration ? Math.min(currentTime / duration * 100, 100) : 0}%` }} />{events.map((event, index) => <button key={index} className={`timeline-marker ${index === selected ? 'selected' : ''}`} style={{ left: `${duration ? Math.min(event.timecode_sec / duration * 100, 100) : 0}%` }} onClick={() => selectEvent(index)} aria-label={`Событие ${index + 1}, ${preciseTime(event.timecode_sec)}`} title={`${event.type} · ${preciseTime(event.timecode_sec)}`} />)}</div><div className="timeline-scale"><span>00:00</span><span>{formatTime(duration / 2)}</span><span>{formatTime(duration)}</span></div></div>
              <p className="source-time-note">Метки относятся к исходной записи. Знаки после запятой не устанавливают точность начала события.</p>

            </section>

            <aside className="events-panel panel" aria-labelledby="events-title"><div className="review-panel-heading"><h3 id="events-title">{events.length ? 'События для проверки' : 'Обзор проверки'}</h3><span className="count-badge">{events.length}</span></div>
              {events.length > 0 ? <><div className="event-list" aria-label="Выберите событие, в хронологическом порядке">{events.map((event, index) => <button key={event.sourceIndex} className={`event-card ${selected === index ? 'active' : ''}`} onClick={() => selectEvent(index)} aria-pressed={selected === index} aria-controls="selected-finding"><span className="event-number">{String(index + 1).padStart(2, '0')}</span><span className="event-summary"><span className="timecode"><Play size={10} weight="fill" />{preciseTime(event.timecode_sec)}</span><strong>{event.type}</strong></span><ChevronRight size={17} /></button>)}</div>{activeEvent && <div className="event-details" id="selected-finding">{activeEvent && <div className="evidence-section"><div className="section-heading"><h3>Кадры события</h3><span>{activeEvent.evidence_frames.length} из исходного видео</span></div><div className="evidence-gallery">{activeEvent.evidence_frames.map((frame, index) => <button className="evidence-card" key={index} aria-label={`Кадр ${index + 1} · ${preciseTime(frame.timestamp_sec)} · Увеличить`} onClick={() => setEnlargedFrame(frame)}><div className="evidence-image"><EvidenceImage frame={frame} alt={`Исходный кадр события ${selected + 1} на отметке ${preciseTime(frame.timestamp_sec)}`} /></div><div className="evidence-caption"><span>Кадр {index + 1}</span><span className="timecode">{preciseTime(frame.timestamp_sec)}</span><Maximize2 size={16} aria-label="Увеличить" /></div></button>)}</div></div>}<div className="detail-section"><span className="detail-label">Наблюдаемые факты</span>{activeMeta?.observed_facts?.length ? <ul className="observed-facts">{activeMeta.observed_facts.map((fact, index) => <li key={index}>{fact}</li>)}</ul> : activeRule?.observation ? <p className="observation">{activeRule.observation}</p> : <p>Отдельный список видимых фактов не передан. Сверьте объяснение модели с исходной записью и кадрами.</p>}</div><div className="detail-section"><span className="detail-label">Описание модели</span><p className="observation">{activeEvent.description}</p></div><div className="rule-card"><div className="rule-icon"><ShieldCheck size={18} /></div><div><span className="detail-label">Источник и пункт</span>{activeRule ? <><h4>{activeRule.title}</h4><p>{activeRule.clause || 'Отдельный пункт не передан сервисом.'}</p>{ruleUrl && <a href={ruleUrl} target="_blank" rel="noopener noreferrer">Открыть источник<ExternalLink size={13} /></a>}</> : <p>Отдельная карточка источника не передана. Проверьте документ и пункт, указанные в описании модели.</p>}</div></div><div className="visibility-note"><CircleAlert size={17} /><div><span className="detail-label">Ограничения видимости</span><p>{[...(activeMeta?.limitations ?? []), ...(activeRule?.limitations ?? [])].join(' ') || 'Отдельные ограничения не переданы. Действия вне кадра, звук и данные банковских систем этой записью не подтверждаются.'}</p></div></div><HumanReviewPanel jobId={jobId} apiKey={key} violationIndex={activeEvent.sourceIndex} /></div>}</> : <div className="no-events"><Illustration className="no-events-art" kind="clear" /><h3>События не выявлены</h3><p>При необходимости просмотрите исходную запись. Отсутствие признаков в видео не подтверждает соблюдение процедур вне кадра.</p></div>}
            </aside>
          </div>}

          {limits.length > 0 && <section className="limitations panel"><CircleAlert size={20} /><div><h3>Ограничения видимости и анализа</h3><ul>{limits.map((limit, index) => <li key={index}>{limit}</li>)}</ul></div></section>}

          <details className="result-diagnostics">
            <summary>Технические сведения</summary>
            <div className="diagnostic-job"><span>Номер задачи</span><code>{jobId}</code></div>
            <section className="result-clocks" aria-label="Время автоматической проверки и просмотра человеком">
              <div className="clock-measurement"><Clock3 size={20} /><div><span>От загрузки до готового результата</span><strong data-e2e-state={failed ? 'failed' : clientDuration !== null ? 'complete' : uploadStartedAt.current === null ? 'restored' : videoError || evidenceError ? 'incomplete' : 'pending'} data-e2e-seconds={clientDuration ?? undefined}>{failed ? failedClientDuration !== null ? `Ошибка через ${elapsedTime(failedClientDuration)}` : 'Проверка не завершена' : clientDuration !== null ? elapsedTime(clientDuration) : uploadStartedAt.current === null ? 'Замер недоступен после восстановления' : videoError || evidenceError ? 'Готовность не подтверждена' : elapsed !== null ? `${elapsedTime(elapsed)} · готовим показ` : '—'}</strong></div></div>
              <div className="clock-measurement"><UserRound size={20} /><div><span>Время просмотра в этой вкладке</span><strong>{humanElapsed === null ? 'Начнётся после подготовки результата' : elapsedTime(humanElapsed)}</strong></div></div>
              <p className="timing-note">{serverElapsed !== null && <span>Интервал сервера после приёма: {elapsedTime(serverElapsed)}. </span>}Полный клиентский замер включает передачу, получение результата, открытие всех JPEG-кадров и подготовку проигрывателя. Время просмотра считается отдельно; это текущий сеанс, а не сохранённая длительность работы эксперта.</p>
            </section>
          </details>
        </>}
      </main>

        <footer className="page-footer" inert={menuOpen}>
          <div className="footer-meta">
            <a href="#main" className="footer-brand" aria-label="FORTE Panopticon"><img className="brand-logo" src="/brand/forte-panopticon.svg" alt="FORTE Panopticon" width="731" height="314" /></a>
            <a href="#main" className="back-top" aria-label="Наверх"><ArrowRight size={20} /></a>
            <div className="footer-socials">
              <p className="footer-follow">Follow us for updates</p>
              <nav className="footer-links" aria-label="Социальные сети">
                <a href="https://github.com/Wezzeso/forte" target="_blank" rel="noopener noreferrer">GitHub<ArrowUpRight size={14} aria-hidden="true" /></a>
                <a href="https://www.linkedin.com/in/timur-ryabov-9886763a4/" target="_blank" rel="noopener noreferrer">LinkedIn<ArrowUpRight size={14} aria-hidden="true" /></a>
                <a href="https://www.instagram.com/wezeso/" target="_blank" rel="noopener noreferrer">Instagram<ArrowUpRight size={14} aria-hidden="true" /></a>
              </nav>
            </div>
            <p className="footer-copyright">All rights reserved © 2026 WEZESO</p>
          </div>
          <div className="footer-signature"><img src="/brand/wezeso-wordmark.png" alt="WEZESO" width="2171" height="724" loading="lazy" decoding="async" /></div>
        </footer>
    </div>

    {keyOpen && <Modal title="Подключить доступ" onClose={() => setKeyOpen(false)}><form className="key-form" onSubmit={saveKey}><Illustration className="access-art" kind="access" /><p>Введите ключ, предоставленный оператором сервиса. Он используется для загрузки видео и просмотра результатов.</p><label htmlFor="api-key">Ключ доступа</label><input id="api-key" type="password" autoComplete="off" value={keyDraft} onChange={event => setKeyDraft(event.target.value)} placeholder={key ? 'Введите ключ для замены' : 'Введите X-API-Key'} spellCheck={false} required /><small><LockKeyhole size={14} />Ключ хранится только в этой вкладке до завершения сеанса.</small><div className="key-form-actions">{key && <button type="button" className="button button-quiet" onClick={() => { setKey(''); remember(KEY_STORAGE, ''); setKeyOpen(false) }}>Отключить доступ</button>}<button className="button button-primary" type="submit" disabled={!keyDraft.trim()}>Подключить<ArrowRight size={17} /></button></div></form></Modal>}

    {enlargedFrame && <Modal title={`Исходный кадр · ${preciseTime(enlargedFrame.timestamp_sec)}`} wide={!portraitVideo} onClose={() => setEnlargedFrame(null)}><EvidenceImage className="enlarged-evidence" frame={enlargedFrame} alt={`Исходный кадр на отметке ${preciseTime(enlargedFrame.timestamp_sec)}`} /><div className="frame-modal-footer"><span>Кадр исходной записи · без графических наложений</span><button className="button button-secondary" onClick={() => { seek(enlargedFrame.timestamp_sec); setEnlargedFrame(null) }}><ArrowLeft size={16} />К этому моменту в видео</button></div></Modal>}
  </div>
}
