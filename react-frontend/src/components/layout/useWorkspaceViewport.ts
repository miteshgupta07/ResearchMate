import { useEffect, useRef } from 'react'

export function useWorkspaceViewport() {
  const workspaceRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const viewport = window.visualViewport
    const workspace = workspaceRef.current
    if (!viewport || !workspace) return
    const resize = () => {
      // Pinch zoom should remain browser-managed rather than resizing the UI.
      if (viewport.scale !== 1) return
      workspace.style.setProperty('--workspace-height', `${viewport.height}px`)
      workspace.style.setProperty('--workspace-top', `${viewport.offsetTop}px`)
      workspace.dataset.compactHeight = String(viewport.height < 480)
    }
    resize()
    viewport.addEventListener('resize', resize)
    viewport.addEventListener('scroll', resize)
    return () => {
      viewport.removeEventListener('resize', resize)
      viewport.removeEventListener('scroll', resize)
      workspace.style.removeProperty('--workspace-height')
      workspace.style.removeProperty('--workspace-top')
      delete workspace.dataset.compactHeight
    }
  }, [])
  return workspaceRef
}
