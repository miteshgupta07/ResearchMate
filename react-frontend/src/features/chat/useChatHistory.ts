import { useEffect, useMemo, useSyncExternalStore } from 'react'
import { createHistoryStore } from './historyStore.ts'

export function useChatHistory(sessionId: string) {
  const store = useMemo(() => createHistoryStore(sessionId), [sessionId])
  const state = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot)

  useEffect(() => {
    void store.load()
    return store.cancel
  }, [store])

  return { ...state, retry: store.load }
}
