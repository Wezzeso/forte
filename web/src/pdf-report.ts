import { jsPDF } from 'jspdf'
import regularFont from './report-fonts/UpMax-Regular.ttf?url'
import boldFont from './report-fonts/UpMax-Bold.ttf?url'
import type { HumanReviews, JobMetadata, JuryJob } from './api'
import { safeSourceUrl } from './api'
import { problemLink, reportTime } from './problem-links'

interface ReportInput {
  job: JuryJob
  metadata: JobMetadata | null
  reviews: HumanReviews | null
  resultHash: string
  sourceIndex?: number
}
const BRAND = '#a31551'
const TEXT = '#242024'
const MUTED = '#696168'
const LEFT = 18
const WIDTH = 174
const BOTTOM = 275

async function fontData(url: string): Promise<string> {
  const response = await fetch(url, { signal: AbortSignal.timeout(30_000) })
  if (!response.ok) throw new Error('Не удалось загрузить шрифт отчёта. Повторите скачивание.')
  const bytes = new Uint8Array(await response.arrayBuffer())
  let binary = ''
  for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192))
  return btoa(binary)
}

function date(value?: string | null): string {
  if (!value || !Number.isFinite(Date.parse(value))) return 'Не передано сервисом'
  return `${new Date(value).toLocaleString('ru-RU', { timeZone: 'Asia/Qyzylorda', day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' })} (UTC+5)`
}

export async function createPdfReport({ job, metadata, reviews, resultHash, sourceIndex }: ReportInput): Promise<Blob> {
  if (job.status !== 'done') throw new Error('Полный результат проверки пока недоступен.')
  const [regular, bold] = await Promise.all([fontData(regularFont), fontData(boldFont)])
  const pdf = new jsPDF({ unit: 'mm', format: 'a4', compress: true, putOnlyUsedFonts: true })
  pdf.addFileToVFS('UpMax-Regular.ttf', regular)
  pdf.addFont('UpMax-Regular.ttf', 'UpMax', 'normal')
  pdf.addFileToVFS('UpMax-Bold.ttf', bold)
  pdf.addFont('UpMax-Bold.ttf', 'UpMax', 'bold')
  pdf.setProperties({ title: 'FORTE Panopticon - Отчёт о проверке видео', author: 'FORTE Panopticon', subject: `Задача ${job.job_id}` })
  pdf.setLanguage('ru')
  let y = 44

  function header() {
    pdf.setFillColor(BRAND); pdf.rect(0, 0, 210, 29, 'F')
    pdf.setFont('UpMax', 'bold'); pdf.setFontSize(17); pdf.setTextColor('#ffffff')
    pdf.text('FORTE / Panopticon', LEFT, 18)
    // Echo the site's zigzag graphic using native PDF paths.
    pdf.setDrawColor('#dc9ab6'); pdf.setLineWidth(1.1)
    for (let x = 148; x < 190; x += 10) { pdf.line(x, 18, x + 5, 11); pdf.line(x + 5, 11, x + 10, 18) }
    y = 44
  }
  function nextPage() { pdf.addPage(); header() }
  function ensure(height: number) { if (y + height > BOTTOM) nextPage() }
  function text(value: string, size = 11, color = TEXT, bold = false, width = WIDTH) {
    pdf.setFont('UpMax', bold ? 'bold' : 'normal'); pdf.setFontSize(size); pdf.setTextColor(color)
    // Normalize source controls; long words and comments wrap without clipping.
    const clean = value.replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g, '').replace(/[\u2011\u2013\u2014]/g, '-')
    const lines: string[] = pdf.splitTextToSize(clean || '-', width)
    const lineHeight = size * .49
    for (const line of lines) {
      ensure(lineHeight)
      // ensure() can install header font/color after a page break.
      pdf.setFont('UpMax', bold ? 'bold' : 'normal'); pdf.setFontSize(size); pdf.setTextColor(color)
      pdf.text(line, LEFT, y)
      y += lineHeight
    }
    y += 2
  }
  function section(label: string, content: string) {
    ensure(20); y += 4; text(label, 10, BRAND, true); text(content)
  }
  function link(label: string, url: string) {
    ensure(10); pdf.setFont('UpMax', 'bold'); pdf.setFontSize(10); pdf.setTextColor(BRAND)
    pdf.textWithLink(label, LEFT, y, { url }); y += 8
  }
  header()
  const all = (job.violations ?? []).map((event, index) => ({ ...event, index })).sort((a, b) => a.timecode_sec - b.timecode_sec || a.index - b.index)
  const included = sourceIndex === undefined ? all : all.filter(event => event.index === sourceIndex)
  if (sourceIndex !== undefined && !included.length) throw new Error('Выбранное событие отсутствует в результате.')

  text('Отчёт о проверке видео', 27, TEXT, true)
  text(sourceIndex === undefined ? 'Полная проверка' : `Выбранное событие · ${reportTime(included[0].timecode_sec)}`, 13, BRAND)
  y += 4
  section('Задача', job.job_id)
  section('Сформирован', date(new Date().toISOString()))
  section('Проверка завершена', date(metadata?.completed_at))
  section('Автоматический результат всей проверки', job.verdict === 'violation' ? `Обнаружены возможные признаки нарушений. Событий: ${all.length}. Окончательный вывод принимает специалист.` : 'Признаки нарушений не обнаружены. Отсутствие признаков в видео не подтверждает соблюдение процедур вне кадра.')
  link('Открыть результат в FORTE Panopticon', problemLink(job.job_id, sourceIndex, included[0]?.timecode_sec))
  section('В этом отчёте', sourceIndex === undefined ? `Событий: ${included.length}. Исходные кадры, описание модели, источники и сохранённые решения человека.` : 'Одно выбранное событие. Остальные события этой проверки в документ не включены.')
  for (const event of included) {
    ensure(16)
    text(`${reportTime(event.timecode_sec)} · ${event.type}`, 11, TEXT, true)
  }
  if (!metadata) section('Дополнительные данные недоступны', 'Не удалось получить метаданные, источники и общие ограничения. Их отсутствие не означает отсутствия ограничений анализа.')
  if (!reviews) section('Сохранённые решения недоступны', 'Сервис не подтвердил решения человека для этой версии результата. Статусы решений, имена и комментарии в отчёт не включены.')
  if (metadata?.limitations?.length) section('Общие ограничения анализа', metadata.limitations.join('\n'))
  section('Как пользоваться отчётом', 'Сверяйте наблюдения с исходной записью. Ссылки открывают нужную задачу и момент видео. Для просмотра в сервисе получателю понадобится ключ доступа; ключ не включён в документ. Метки времени не устанавливают точность начала события.')

  for (const event of included) {
    nextPage()
    text(`Событие ${all.findIndex(item => item.index === event.index) + 1} / ${all.length}`, 11, BRAND, true)
    text(event.type, 21, TEXT, true)
    text(`Исходная запись · ${reportTime(event.timecode_sec)}`, 12, MUTED)
    link('Открыть это событие и момент видео', problemLink(job.job_id, event.index, event.timecode_sec))

    const info = metadata?.events?.[event.index]
    const rule = info?.rule ?? metadata?.rules?.[event.index]
    section('Наблюдаемые факты', info?.observed_facts?.length ? info.observed_facts.join('\n') : rule?.observation || 'Отдельный список видимых фактов не передан. Сверьте описание модели с записью и кадрами.')
    section('Описание модели', event.description)
    section('Источник и пункт', rule ? `${rule.title}\n${rule.clause || 'Отдельный пункт не передан сервисом.'}` : 'Отдельная карточка источника не передана. Проверьте документ и пункт в описании модели.')
    const ruleUrl = safeSourceUrl(rule?.url)
    if (ruleUrl) link('Открыть источник', ruleUrl)
    const saved = reviews?.violations.find(item => item.violation_index === event.index)
    const review = saved?.review
    section('Сохранённое решение человека', !review ? 'Недоступно для этой версии результата.' : review.state === 'confirmed' ? 'Признак подтверждён человеком' : review.state === 'rejected' ? 'Признак отклонён человеком' : 'Ожидает решения человека')
    if (review && review.state !== 'pending') {
      text(`Проверяющий: ${review.claimed_reviewer_name || 'Не указан'}. Имя указано пользователем, без проверки личности.`, 10, MUTED)
      text(`Сохранено: ${date(review.reviewed_at)}. Версия решения: ${review.revision}.`, 10, MUTED)
      if (review.comment) section('Сохранённый комментарий', review.comment)
    }
    section('Ограничения видимости', [...(info?.limitations ?? []), ...(rule?.limitations ?? [])].join('\n') || 'Отдельные ограничения не переданы. Действия вне кадра, звук и данные банковских систем этой записью не подтверждаются.')

    const gap = 5
    const cellWidth = (WIDTH - gap * (event.evidence_frames.length - 1)) / event.evidence_frames.length
    const maxImageHeight = Math.max(30, Math.min(55, BOTTOM - y - 35))
    const frames = event.evidence_frames.map(frame => {
      const data = `data:image/jpeg;base64,${frame.image_base64}`
      const image = pdf.getImageProperties(data)
      const width = Math.min(cellWidth, maxImageHeight * image.width / image.height)
      return { ...frame, data, width, height: width * image.height / image.width }
    })
    const height = Math.max(...frames.map(frame => frame.height))
    ensure(height + 28)
    y += 5; text('Кадры исходной записи', 10, BRAND, true)
    frames.forEach((frame, i) => {
      const x = LEFT + i * (cellWidth + gap)
      pdf.setFillColor('#f2f2f2'); pdf.rect(x, y, cellWidth, height, 'F')
      pdf.addImage(frame.data, 'JPEG', x + (cellWidth - frame.width) / 2, y + (height - frame.height) / 2, frame.width, frame.height)
      pdf.setFont('UpMax', 'normal'); pdf.setFontSize(9); pdf.setTextColor(MUTED)
      pdf.text(`Кадр ${i + 1} · ${reportTime(frame.timestamp_sec)}`, x, y + height + 6)
    })
    y += height + 15
    text('Без графических наложений. Пропорции исходных кадров сохранены.', 9, MUTED)
  }

  nextPage()
  text('Привязка отчёта', 22, TEXT, true)
  section('SHA-256 результата модели', resultHash)
  if (reviews) section('SHA-256 исходного видео', reviews.source_sha256)
  for (const event of included) {
    const binding = reviews?.violations.find(item => item.violation_index === event.index)
    if (binding) section(`SHA-256 доказательств · ${reportTime(event.timecode_sec)}`, binding.evidence_sha256)
  }
  section('Данные документа', 'Отчёт содержит снимок автоматического результата и доступных сохранённых решений на момент формирования. Последующие решения человека могут измениться. Черновики формы не включены. Решения человека не изменяют исходный вердикт модели.')
  text('FORTE Panopticon · Прототип для дататона', 10, MUTED)

  const count = pdf.getNumberOfPages()
  for (let page = 1; page <= count; page++) {
    pdf.setPage(page); pdf.setDrawColor('#ded8de'); pdf.setLineWidth(.25); pdf.line(LEFT, 283, 192, 283)
    pdf.setFont('UpMax', 'normal'); pdf.setFontSize(8); pdf.setTextColor(MUTED)
    pdf.text('FORTE Panopticon · Отчёт о проверке', LEFT, 289)
    pdf.text(`${page} / ${count}`, 192, 289, { align: 'right' })
  }
  return pdf.output('blob')
}
