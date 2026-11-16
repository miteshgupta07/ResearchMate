import { useLayoutEffect, useRef } from 'react'
import { createScrollFollower } from './chatLayout'

export function useChatScroll(update: unknown) {
  const viewportRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const followerRef = useRef<ReturnType<typeof createScrollFollower> | null>(null)
  useLayoutEffect(() => {
    const follower = createScrollFollower(() => viewportRef.current)
    followerRef.current = follower
    follower.update()
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(follower.update)
    if (contentRef.current) observer?.observe(contentRef.current)
    if (viewportRef.current) observer?.observe(viewportRef.current)
    return () => { observer?.disconnect(); followerRef.current = null }
  }, [])
  useLayoutEffect(() => { followerRef.current?.update() }, [update])

  return {
    viewportRef, contentRef,
    followLatest: () => { followerRef.current?.followLatest() },
    onScroll: () => { followerRef.current?.onScroll() },
  }
}
