export interface ProblemTarget { jobId: string; eventIndex: number | null; seconds: number | null }

export function readProblemTarget(): ProblemTarget | null {
  const query = new URLSearchParams(window.location.search)
  const jobId = query.get('job')
  if (!jobId || !/^[A-Za-z0-9_-]{1,128}$/.test(jobId)) return null
  const event = query.get('event')
  const time = query.get('t')
  return {
    jobId,
    eventIndex: event !== null && /^\d+$/.test(event) && Number.isSafeInteger(Number(event)) ? Number(event) : null,
    seconds: time !== null && /^\d+(\.\d{1,3})?$/.test(time) && Number.isFinite(Number(time)) ? Number(time) : null,
  }
}

// Only task identity and source time travel in a link. Never copy credentials,
// an operator API origin, or unrelated query parameters from the current URL.
export function problemLink(jobId: string, eventIndex?: number, seconds?: number): string {
  const url = new URL(window.location.pathname, window.location.origin)
  url.searchParams.set('job', jobId)
  if (eventIndex !== undefined) url.searchParams.set('event', String(eventIndex))
  if (seconds !== undefined) url.searchParams.set('t', String(Math.round(seconds * 1000) / 1000))
  return url.href
}

export function reportTime(seconds: number): string {
  const ms = Math.round(Math.max(0, seconds) * 1000)
  return `${String(Math.floor(ms / 60000)).padStart(2, '0')}:${String(Math.floor(ms / 1000) % 60).padStart(2, '0')}.${String(ms % 1000).padStart(3, '0')}`
}
