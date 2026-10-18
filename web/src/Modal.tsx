import { useEffect, useRef } from 'react'
import type { ReactNode } from 'react'
import { XIcon as X } from '@phosphor-icons/react'

export default function Modal({ title, onClose, children, wide = false }: { title: string; onClose: () => void; children: ReactNode; wide?: boolean }) {
  const dialogRef = useRef<HTMLDivElement>(null)
  const close = useRef(onClose)
  close.current = onClose
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null
    const dialog = dialogRef.current
    const selector = 'button:not(:disabled), input:not(:disabled), textarea:not(:disabled), select:not(:disabled), a[href], [tabindex="0"]'
    dialog?.querySelector<HTMLElement>(selector)?.focus()
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close.current()
      if (event.key !== 'Tab' || !dialog) return
      const focusable = [...dialog.querySelectorAll<HTMLElement>(selector)]
      const start = focusable[0]
      const end = focusable[focusable.length - 1]
      if (!start) { event.preventDefault(); dialog.focus(); return }
      if (event.shiftKey && document.activeElement === start) { event.preventDefault(); end?.focus() }
      if (!event.shiftKey && document.activeElement === end) { event.preventDefault(); start?.focus() }
    }
    document.addEventListener('keydown', onKey)
    const oldOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = oldOverflow
      if (previous?.isConnected) previous.focus()
      else [...document.querySelectorAll<HTMLButtonElement>('.key-button, .menu-toggle')].find(button => button.getClientRects().length > 0)?.focus()
    }
  }, [])
  return <div className="modal-overlay" onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}>
    <div className={`modal ${wide ? 'modal-wide' : ''}`} role="dialog" aria-modal="true" aria-label={title} tabIndex={-1} ref={dialogRef}>
      <div className="modal-heading"><h2>{title}</h2><button className="icon-button" onClick={onClose} aria-label="Закрыть"><X size={20} /></button></div>
      {children}
    </div>
  </div>
}
