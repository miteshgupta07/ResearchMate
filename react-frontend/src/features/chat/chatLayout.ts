export function resizeChatInput(textarea: HTMLTextAreaElement): void {
  // Reset first so deleting text can shrink the composer as well as grow it.
  // CSS applies min/max height, including viewport-dependent math expressions.
  textarea.style.height = '0px'
  textarea.style.height = `${textarea.scrollHeight}px`
  textarea.style.overflowY = textarea.scrollHeight > textarea.clientHeight ? 'auto' : 'hidden'
}

export function shouldSendOnEnter(event: { key: string; shiftKey: boolean; isComposing: boolean; keyCode: number }): boolean {
  return event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229
}

type ScrollViewport = Pick<HTMLElement, 'scrollTop' | 'scrollHeight' | 'clientHeight'>

export function createScrollFollower(getViewport: () => ScrollViewport | null) {
  let following = true
  return {
    onScroll: () => {
      const viewport = getViewport()
      if (viewport) following = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight <= 80
    },
    update: () => {
      const viewport = getViewport()
      if (viewport && following) viewport.scrollTop = viewport.scrollHeight
    },
    followLatest: () => { following = true },
  }
}
