import { useEffect, useMemo, useSyncExternalStore } from 'react'
import { createChatStore } from './chatStore.ts'

export function useChat(sessionId: string) {
  const store = useMemo(() => createChatStore(sessionId), [sessionId])
  const state = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot)
  useEffect(() => {
    void store.load()
    return store.cancel
  }, [store])
  return { ...state, submit: store.submit, retry: store.load }
}
