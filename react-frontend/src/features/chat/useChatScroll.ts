import { useLayoutEffect, useRef } from 'react'

export function useChatScroll(update: unknown) {
  const viewportRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const following = useRef(true)

  const scrollIfFollowing = () => {
    const viewport = viewportRef.current
    if (viewport && following.current) viewport.scrollTop = viewport.scrollHeight
  }
  useLayoutEffect(scrollIfFollowing, [update])
  useLayoutEffect(() => {
    if (typeof ResizeObserver === 'undefined' || !contentRef.current) return
    const observer = new ResizeObserver(scrollIfFollowing)
    observer.observe(contentRef.current)
    return () => observer.disconnect()
  }, [])

  return {
    viewportRef, contentRef,
    followLatest: () => { following.current = true },
    onScroll: () => {
      const viewport = viewportRef.current
      if (viewport) following.current = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 80
    },
  }
}
