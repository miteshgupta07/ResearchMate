export function dropdownKey(key: string, open: boolean, active: number, count: number) {
  if (key === 'Escape' || key === 'Tab') return { type: open ? 'close' : 'none', index: active } as const
  if (key === 'Enter' || key === ' ') return { type: open ? 'select' : 'open', index: active } as const
  if (key === 'ArrowDown' || key === 'ArrowUp') {
    const offset = key === 'ArrowDown' ? 1 : -1
    return { type: open ? 'move' : 'open', index: open ? (active + offset + count) % count : active } as const
  }
  if (open && (key === 'Home' || key === 'End')) return { type: 'move', index: key === 'Home' ? 0 : count - 1 } as const
  return { type: 'none', index: active } as const
}

export function dropdownPosition(anchor: { left: number; top: number; bottom: number; width: number }, viewport: { left: number; top: number; width: number; height: number }, rows: number) {
  const gap = 6
  const edge = 8
  const desiredHeight = Math.min(rows * 44 + 8, 272)
  const below = Math.max(0, viewport.top + viewport.height - anchor.bottom - gap - edge)
  const above = Math.max(0, anchor.top - viewport.top - gap - edge)
  const upward = below < desiredHeight && above > below
  const maxHeight = Math.min(desiredHeight, upward ? above : below)
  const width = Math.min(anchor.width, viewport.width - edge * 2)
  const left = Math.max(viewport.left + edge, Math.min(anchor.left, viewport.left + viewport.width - width - edge))
  return { left, width, maxHeight, top: upward ? anchor.top - gap - maxHeight : anchor.bottom + gap }
}
