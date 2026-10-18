// The operator controls this same-origin JSON file. No browser storage, URL
// parameter, form field, or global variable can select the API destination.
let apiBaseUrl = ''

export async function initializeRuntimeConfig(): Promise<void> {
  apiBaseUrl = ''
  try {
    const response = await fetch('/runtime-config.json', {
      cache: 'no-store', credentials: 'same-origin', redirect: 'error',
      signal: AbortSignal.timeout(1500),
    })
    if (!response.ok || !response.headers.get('content-type')?.includes('application/json')) return
    const config: unknown = await response.json()
    if (!config || typeof config !== 'object' || !('apiBaseUrl' in config) || typeof config.apiBaseUrl !== 'string' || config.apiBaseUrl.length > 2048) return
    const target = new URL(config.apiBaseUrl)
    if (target.protocol !== 'https:' || target.username || target.password || target.search || target.hash || target.pathname !== '/') return
    apiBaseUrl = target.origin
  } catch {
    // An absent, unreachable or invalid optional configuration uses same origin.
  }
}

export function apiUrl(path: string): string {
  if (!path.startsWith('/') || path.startsWith('//')) throw new Error('Некорректный путь сервиса.')
  return `${apiBaseUrl}${path}`
}
