import { useCallback, useEffect, useRef, useState } from 'react'
import { Outlet, useLocation } from 'react-router'
import { navigation } from '../../app/navigation'
import { Header } from './Header'
import { Sidebar } from './Sidebar'
import { ModeIndicator } from '../../features/settings/ModeIndicator'
import { useWorkspaceViewport } from './useWorkspaceViewport'
import { OpeningSplash } from '../ui/OpeningSplash'

export function AppLayout() {
  const workspaceRef = useWorkspaceViewport()
  const [collapsed, setCollapsed] = useState(true)
  const [splashComplete, setSplashComplete] = useState(false)
  const autoExpandTimer = useRef<number | undefined>(undefined)
  const autoExpandCancelled = useRef(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const drawerRef = useRef<HTMLDialogElement>(null)
  const mainRef = useRef<HTMLElement>(null)
  const { pathname } = useLocation()
  const previousPath = useRef(pathname)
  const title = navigation.find((item) => item.path === pathname)?.label ?? 'ResearchMate'
  const isChat = pathname === '/'

  const onSplashComplete = useCallback(() => setSplashComplete(true), [])
  const cancelAutoExpansion = useCallback(() => {
    autoExpandCancelled.current = true
    window.clearTimeout(autoExpandTimer.current)
  }, [])
  const toggleSidebar = () => {
    cancelAutoExpansion()
    setCollapsed((value) => !value)
  }
  const openMobileNavigation = () => {
    cancelAutoExpansion()
    setMobileOpen(true)
  }

  useEffect(() => {
    if (!splashComplete || autoExpandCancelled.current) return
    const desktop = window.matchMedia('(min-width: 768px)')
    if (!desktop.matches) return

    autoExpandTimer.current = window.setTimeout(() => {
      if (desktop.matches && !autoExpandCancelled.current) setCollapsed(false)
    }, 2000)
    const cancelOnMobile = () => {
      if (!desktop.matches) cancelAutoExpansion()
    }
    desktop.addEventListener('change', cancelOnMobile)
    return () => {
      window.clearTimeout(autoExpandTimer.current)
      desktop.removeEventListener('change', cancelOnMobile)
    }
  }, [splashComplete, cancelAutoExpansion])

  useEffect(() => {
    document.title = `${title} · ResearchMate`
    if (previousPath.current !== pathname) {
      mainRef.current?.focus()
      mainRef.current?.scrollTo(0, 0)
      previousPath.current = pathname
    }
  }, [pathname, title])

  useEffect(() => {
    const drawer = drawerRef.current
    if (!drawer || !mobileOpen) return
    // Native dialog provides focus containment, Escape, and background inertness.
    drawer.showModal()
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const desktop = window.matchMedia('(min-width: 768px)')
    const closeOnDesktop = () => {
      if (desktop.matches) setMobileOpen(false)
    }
    closeOnDesktop()
    desktop.addEventListener('change', closeOnDesktop)
    return () => {
      desktop.removeEventListener('change', closeOnDesktop)
      document.body.style.overflow = previousOverflow
      drawer.close()
    }
  }, [mobileOpen])

  return (
    <div ref={workspaceRef} className="group/workspace fixed inset-x-0 top-(--workspace-top,0px) flex h-(--workspace-height,100dvh) min-h-0 overflow-hidden bg-canvas">
      <OpeningSplash onComplete={onSplashComplete} />
      <a href="#main-content" className="sr-only fixed top-3 left-3 z-50 rounded-lg bg-ink px-4 py-3 text-sm font-medium text-sidebar focus:not-sr-only">Skip to content</a>
      <aside aria-label="Sidebar" className={`hidden shrink-0 overflow-hidden border-r border-line/50 bg-sidebar transition-[width] duration-300 ease-in-out motion-reduce:transition-none md:block ${collapsed ? 'w-20' : 'w-64'}`}>
        <Sidebar collapsed={collapsed} onToggle={toggleSidebar} />
      </aside>
      <dialog ref={drawerRef} id="mobile-sidebar" aria-label="Navigation" onClose={() => setMobileOpen(false)} onClick={(event) => { if (event.target === event.currentTarget) setMobileOpen(false) }} className="fixed inset-y-0 left-0 m-0 h-dvh max-h-none w-72 max-w-[85vw] border-0 bg-sidebar p-0 text-ink shadow-2xl backdrop:bg-black/60">
        <Sidebar mobile onToggle={() => setMobileOpen(false)} onNavigate={() => setMobileOpen(false)} />
      </dialog>
      <div className="flex min-w-0 flex-1 flex-col">
        {!isChat && <Header title={title} mobileOpen={mobileOpen} onOpenNavigation={openMobileNavigation} />}
        <main ref={mainRef} id="main-content" tabIndex={-1} aria-label={title} className={`flex min-h-0 flex-1 flex-col outline-none focus-visible:outline-none ${isChat ? 'overflow-hidden' : 'overflow-y-auto'}`}>
          {!isChat && <div className="shrink-0"><ModeIndicator /></div>}
          <Outlet context={{ chatHeader: <><Header title={title} mobileOpen={mobileOpen} onOpenNavigation={openMobileNavigation} /><ModeIndicator /></> }} />
          {!isChat && <footer className="mt-auto shrink-0 px-6 py-6 text-center text-[11px] tracking-wide text-muted">ResearchMate <span aria-hidden="true" className="px-1.5">·</span> A space to explore ideas</footer>}
        </main>
      </div>
    </div>
  )
}
