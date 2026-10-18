/**
 * Local synthetic design preview. Run Vite on 127.0.0.1:5173, then:
 *   node qa/preview-server.mjs
 * Open http://127.0.0.1:5174/?state=positive (or negative, failed, processing).
 * The root URL shows the normal landing state. This file is never imported by
 * the application, and all session seeding and API mocks stay in this QA proxy.
 */
import http from 'node:http'
import net from 'node:net'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const qaDirectory = path.dirname(fileURLToPath(import.meta.url))
const positive = JSON.parse(fs.readFileSync(path.resolve(qaDirectory, '../../contracts/fixtures/positive.json'), 'utf8'))
const videoPath = path.join(qaDirectory, 'fixtures/review-synthetic.mov')
const testKey = 'synthetic-browser-only-key'
const resultHash = 'a'.repeat(64)
const sourceHash = 'b'.repeat(64)
const modes = new Set(['positive', 'negative', 'failed', 'processing'])
const evidenceHash = index => String(index + 1).repeat(64)
const pendingReview = index => ({ job_id: positive.job_id, result_sha256: resultHash, violation_index: index, evidence_sha256: evidenceHash(index), state: 'pending', revision: 0, claimed_reviewer_name: null, comment: '', reviewed_at: null, reviewer_identity_verified: false })
const ledgers = new Map()
const histories = new Map()

function syntheticPositive() {
  const base = positive.violations[0]
  return { ...positive, violations: [
    { ...base, type: 'Синтетическое событие для проверки интерфейса', description: 'Это искусственный пример интерфейса, а не наблюдение из реальной записи. Синтетический документ, раздел TEST-1. Не является нормативным или дисциплинарным выводом.' },
    { ...base, type: 'Второе отдельное синтетическое событие', timecode_sec: 7.2 },
  ] }
}

function ledger(mode) {
  if (!ledgers.has(mode)) ledgers.set(mode, { job_id: positive.job_id, result_sha256: resultHash, source_sha256: sourceHash, reviewer_identity_verified: false, violations: [0, 1].map(index => ({ violation_index: index, evidence_sha256: evidenceHash(index), review: pendingReview(index) })) })
  return ledgers.get(mode)
}

function modeFromRequest(request) {
  const referringPage = new URL(request.headers.referer || '/', 'http://127.0.0.1:5174')
  const mode = referringPage.searchParams.get('state')
  return modes.has(mode) ? mode : 'positive'
}

function json(response, body, status = 200) {
  response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' })
  response.end(JSON.stringify(body))
}

async function readJson(request) {
  const chunks = []
  let size = 0
  for await (const chunk of request) {
    size += chunk.length
    if (size > 1024 * 1024) throw new Error('QA request exceeds 1 MiB')
    chunks.push(chunk)
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8'))
}

async function mockApi(request, response, url) {
  if (url.pathname === '/runtime-config.json') return json(response, { detail: 'optional_config_missing' }, 404)
  if (url.pathname === '/readyz') return json(response, { status: 'ready', components: { database: true, storage: true, model: true, worker: true } })
  if (request.headers['x-api-key'] !== testKey) return json(response, { detail: 'synthetic_preview_key_required' }, 401)
  const mode = modeFromRequest(request)
  if (url.pathname === '/api/ui/config') return json(response, { max_upload_bytes: 256 * 1024 * 1024, accepted_suffixes: ['.mov'] })
  if (url.pathname === '/api/v1/analyze' && request.method === 'POST') {
    request.resume()
    return json(response, { job_id: positive.job_id }, 202)
  }
  const resultPath = `/api/v1/jobs/${positive.job_id}`
  const metadataPath = `/api/ui/jobs/${positive.job_id}`
  if (url.pathname === resultPath) return json(response, mode === 'positive' ? syntheticPositive() : mode === 'negative' ? { job_id: positive.job_id, status: 'done', verdict: 'no_violation', violations: [] } : { job_id: positive.job_id, status: mode })
  if (url.pathname === metadataPath) return json(response, {
    job_id: positive.job_id, status: mode === 'processing' ? 'processing' : mode === 'failed' ? 'failed' : 'done',
    stage: mode === 'processing' ? 'screening' : mode === 'failed' ? 'failed' : 'done',
    created_at: new Date(Date.now() - 15_000).toISOString(), elapsed_sec: 12.5,
    limitations: mode === 'positive' ? ['Синтетический пример ограничения видимости; реальные записи не используются.'] : [],
    events: [{ rule: { title: 'Синтетический документ для проверки источников', clause: 'Раздел TEST-1 · пример, не нормативный акт', url: 'https://example.com/synthetic' } }],
    error_code: mode === 'failed' ? 'synthetic_test_failure' : null,
  })
  if (url.pathname === `${metadataPath}/video`) {
    response.writeHead(200, { 'Content-Type': 'video/quicktime', 'Content-Length': fs.statSync(videoPath).size, 'Cache-Control': 'no-store' })
    return fs.createReadStream(videoPath).pipe(response)
  }
  if (url.pathname === `${metadataPath}/reviews`) return json(response, ledger(mode))
  const historyMatch = url.pathname.match(/\/reviews\/(\d+)\/history$/)
  if (historyMatch) {
    const index = Number(historyMatch[1])
    const entries = (histories.get(mode) || []).filter(record => record.violation_index === index && record.revision > Number(url.searchParams.get('after_revision') || 0))
    return json(response, { job_id: positive.job_id, result_sha256: resultHash, violation_index: index, entries, next_after_revision: entries.at(-1)?.revision || 0, has_more: false })
  }
  const reviewMatch = url.pathname.match(/\/reviews\/(\d+)$/)
  if (reviewMatch && request.method === 'POST') {
    const index = Number(reviewMatch[1])
    const item = ledger(mode).violations[index]
    if (!item) return json(response, { detail: 'synthetic_finding_not_found' }, 404)
    const input = await readJson(request)
    if (input.expected_revision !== item.review.revision || input.result_sha256 !== resultHash || input.evidence_sha256 !== item.evidence_sha256) return json(response, { detail: 'review_conflict', current_revision: item.review.revision, current_result_sha256: resultHash }, 409)
    item.review = { ...item.review, state: input.decision === 'confirm' ? 'confirmed' : 'rejected', revision: item.review.revision + 1, claimed_reviewer_name: input.claimed_reviewer_name, comment: input.comment, reviewed_at: new Date().toISOString() }
    histories.set(mode, [...(histories.get(mode) || []), { ...item.review }])
    return json(response, item.review)
  }
  return json(response, { detail: 'synthetic_preview_route_not_found' }, 404)
}

function proxyVite(request, response, url) {
  const requestHeaders = { ...request.headers, host: '127.0.0.1:5173', 'accept-encoding': 'identity' }
  delete requestHeaders['if-none-match']
  delete requestHeaders['if-modified-since']
  const upstream = http.request({ host: '127.0.0.1', port: 5173, path: request.url, method: request.method, headers: requestHeaders }, viteResponse => {
    const html = (viteResponse.headers['content-type'] || '').includes('text/html')
    if (!html) {
      response.writeHead(viteResponse.statusCode || 502, viteResponse.headers)
      return viteResponse.pipe(response)
    }
    const chunks = []
    viteResponse.on('data', chunk => chunks.push(chunk))
    viteResponse.on('end', () => {
      const mode = url.searchParams.get('state')
      const seed = modes.has(mode)
        ? `sessionStorage.setItem('forte.session.key', ${JSON.stringify(testKey)}); sessionStorage.setItem('forte.session.job', ${JSON.stringify(positive.job_id)});`
        : "sessionStorage.removeItem('forte.session.key'); sessionStorage.removeItem('forte.session.job');"
      const injected = `<script>/* Local QA preview only. */ ${seed}</script>`
      const body = Buffer.from(Buffer.concat(chunks).toString('utf8').replace('<head>', `<head>${injected}`))
      const headers = { ...viteResponse.headers, 'content-length': body.length, 'cache-control': 'no-store' }
      delete headers['content-encoding']
      delete headers['transfer-encoding']
      delete headers.etag
      response.writeHead(viteResponse.statusCode || 200, headers)
      response.end(body)
    })
  })
  upstream.on('error', error => {
    response.writeHead(502, { 'Content-Type': 'text/plain; charset=utf-8' })
    response.end(`QA proxy cannot reach Vite on 127.0.0.1:5173. Start npm run dev -- --port 5173.\n${error.message}`)
  })
  request.pipe(upstream)
}

const server = http.createServer(async (request, response) => {
  const url = new URL(request.url || '/', 'http://127.0.0.1:5174')
  try {
    if (url.pathname.startsWith('/api/') || url.pathname === '/readyz' || url.pathname === '/runtime-config.json') await mockApi(request, response, url)
    else proxyVite(request, response, url)
  } catch (error) {
    if (!response.headersSent) json(response, { detail: error.message }, 500)
    else response.destroy(error)
  }
})

// Keep Vite's development HMR on the local QA origin.
server.on('upgrade', (request, socket, head) => {
  const upstream = net.connect(5173, '127.0.0.1', () => {
    upstream.write(`${request.method} ${request.url} HTTP/${request.httpVersion}\r\n${request.rawHeaders.map((header, index) => index % 2 === 0 ? `${header}: ` : `${header}\r\n`).join('')}\r\n`)
    if (head.length) upstream.write(head)
    socket.pipe(upstream).pipe(socket)
  })
  upstream.on('error', () => socket.destroy())
  socket.on('error', () => upstream.destroy())
})

server.listen(5174, '127.0.0.1', () => process.stdout.write('Synthetic QA preview: http://127.0.0.1:5174/?state=positive\n'))
