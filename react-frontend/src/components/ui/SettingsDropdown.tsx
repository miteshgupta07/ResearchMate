import { useEffect, useId, useRef, useState } from 'react'
import { dropdownKey, dropdownPosition } from './dropdownBehavior'

interface SettingsDropdownProps<T extends string> {
  id: string
  labelId: string
  descriptionId?: string
  value: T
  options: readonly T[]
  onChange: (value: T) => void
}

export function SettingsDropdown<T extends string>({ id, labelId, descriptionId, value, options, onChange }: SettingsDropdownProps<T>) {
  const menuId = useId()
  const triggerRef = useRef<HTMLButtonElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const searchRef = useRef({ text: '', time: 0 })

  const position = () => {
    const trigger = triggerRef.current
    const menu = menuRef.current
    if (!trigger || !menu) return
    const viewport = window.visualViewport
    const rect = dropdownPosition(trigger.getBoundingClientRect(), {
      left: viewport?.offsetLeft ?? 0, top: viewport?.offsetTop ?? 0,
      width: viewport?.width ?? window.innerWidth, height: viewport?.height ?? window.innerHeight,
    }, options.length)
    Object.assign(menu.style, { left: `${rect.left}px`, top: `${rect.top}px`, width: `${rect.width}px`, maxHeight: `${rect.maxHeight}px` })
  }

  const close = () => { menuRef.current?.hidePopover(); setOpen(false) }
  const show = () => {
    setActive(Math.max(0, options.indexOf(value)))
    searchRef.current = { text: '', time: 0 }
    position()
    menuRef.current?.showPopover()
    setOpen(true)
  }
  const choose = (index: number) => {
    const option = options[index]
    if (option === undefined) return
    onChange(option)
    close()
    triggerRef.current?.focus({ preventScroll: true })
  }

  useEffect(() => {
    const menu = menuRef.current
    if (!menu) return
    const onToggle = (event: Event) => setOpen((event as ToggleEvent).newState === 'open')
    menu.addEventListener('toggle', onToggle)
    return () => { menu.removeEventListener('toggle', onToggle); menu.hidePopover() }
  }, [])

  useEffect(() => {
    if (!open) return
    // Close on surrounding scroll/resize so the menu never floats away from its trigger.
    const dismiss = (event: Event) => {
      if (event.target instanceof Node && menuRef.current?.contains(event.target)) return
      menuRef.current?.hidePopover()
      setOpen(false)
    }
    window.addEventListener('scroll', dismiss, true)
    window.addEventListener('resize', dismiss)
    window.visualViewport?.addEventListener('resize', dismiss)
    return () => {
      window.removeEventListener('scroll', dismiss, true)
      window.removeEventListener('resize', dismiss)
      window.visualViewport?.removeEventListener('resize', dismiss)
    }
  }, [open])

  useEffect(() => {
    const menu = menuRef.current
    const option = menu?.querySelector<HTMLElement>(`[data-option-index="${active}"]`)
    if (!open || !menu || !option) return
    if (option.offsetTop < menu.scrollTop) menu.scrollTop = option.offsetTop
    else if (option.offsetTop + option.offsetHeight > menu.scrollTop + menu.clientHeight) menu.scrollTop = option.offsetTop + option.offsetHeight - menu.clientHeight
  }, [active, open])

  return (
    <>
      <button ref={triggerRef} id={id} type="button" role="combobox" aria-haspopup="listbox" aria-expanded={open} aria-controls={menuId} aria-labelledby={labelId} aria-describedby={descriptionId} aria-activedescendant={open ? `${menuId}-${active}` : undefined} onClick={() => open ? close() : show()} onKeyDown={(event) => {
        const selectedIndex = Math.max(0, options.indexOf(value))
        const action = dropdownKey(event.key, open, open ? active : selectedIndex, options.length)
        if (action.type !== 'none') {
          if (event.key !== 'Tab') { event.preventDefault(); event.stopPropagation() }
          if (action.type === 'open') show()
          else if (action.type === 'close') close()
          else if (action.type === 'select') choose(active)
          else setActive(action.index)
        } else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
          event.preventDefault()
          const now = Date.now()
          const text = (now - searchRef.current.time < 600 ? searchRef.current.text : '') + event.key.toLowerCase()
          const index = options.findIndex((option) => option.toLowerCase().startsWith(text))
          if (!open) show()
          searchRef.current = { text, time: now }
          if (index >= 0) setActive(index)
        }
      }} className={`settings-dropdown-trigger mt-2 flex min-h-11 w-full items-center justify-between gap-2 rounded-xl border bg-surface px-3 py-2 text-left text-sm text-ink transition-colors ${open ? 'border-accent/60' : 'border-line hover:border-muted/60'}`}>
        <span className="truncate">{value}</span><span aria-hidden="true" className={`text-muted transition-transform ${open ? 'rotate-180' : ''}`}>⌄</span>
      </button>
      <div ref={menuRef} id={menuId} popover="auto" role="listbox" aria-labelledby={labelId} className="settings-dropdown-menu chat-scrollbar fixed inset-auto m-0 overflow-y-auto overscroll-contain rounded-xl border border-line bg-surface p-1 text-sm text-ink shadow-xl">
        {options.map((option, index) => (
          <div key={option} id={`${menuId}-${index}`} data-option-index={index} role="option" aria-selected={option === value} onPointerMove={() => setActive(index)} onPointerDown={(event) => event.preventDefault()} onClick={() => choose(index)} className={`flex min-h-11 cursor-pointer items-center justify-between gap-2 rounded-lg px-3 ${active === index ? 'bg-accent/10 text-ink' : 'text-muted'} ${option === value ? 'font-medium text-ink' : ''}`}>
            <span>{option}</span>{option === value && <span aria-hidden="true" className="text-accent">✓</span>}
          </div>
        ))}
      </div>
    </>
  )
}
