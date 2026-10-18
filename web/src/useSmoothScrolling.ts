import { useCallback, useEffect, useRef } from 'react'
import Lenis from 'lenis'
import 'lenis/dist/lenis.css'

export default function useSmoothScrolling() {
  const scroller = useRef<Lenis | null>(null)

  useEffect(() => {
    const lenis = new Lenis({
      autoRaf: true,
      lerp: 0.14,
      smoothWheel: true,
      syncTouch: false,
      respectReducedMotion: true,
      stopInertiaOnNavigate: true,
      anchors: { duration: 0.6, lerp: 0, easing: t => 1 - (1 - t) ** 3 },
      prevent: node => node.matches('[aria-modal="true"], .mobile-navigation, textarea, video'),
    })
    scroller.current = lenis

    // Existing dialogs and navigation lock the body; stop pending wheel inertia too.
    const syncLock = () => {
      if (document.body.style.overflow === 'hidden') lenis.stop()
      else lenis.start()
    }
    const observer = new MutationObserver(syncLock)
    observer.observe(document.body, { attributes: true, attributeFilter: ['style'] })
    syncLock()
    return () => {
      observer.disconnect()
      lenis.destroy()
      scroller.current = null
    }
  }, [])

  return useCallback(() => {
    // Replacing a long page invalidates both its dimensions and any scroll target.
    if (scroller.current) {
      scroller.current.resize()
      scroller.current.scrollTo(0, { immediate: true, force: true })
    } else window.scrollTo({ top: 0, behavior: 'instant' })
  }, [])
}
