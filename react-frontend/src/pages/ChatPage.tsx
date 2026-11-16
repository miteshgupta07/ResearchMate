import { useSession } from '../app/session/SessionContext'
import { LANGUAGE_CODES } from '../features/settings/model'
import { useChatHistory } from '../features/chat/useChatHistory'
import { ChatHistoryView } from '../features/chat/ChatHistoryView'

export function ChatPage() {
  const { session_id, settings, welcomeMessage } = useSession()
  const history = useChatHistory(session_id)
  return <ChatHistoryView history={history} onRetry={history.retry} welcomeMessage={welcomeMessage} languageCode={LANGUAGE_CODES[settings.language]} />
}
