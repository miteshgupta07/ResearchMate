import { PagePlaceholder } from '../components/ui/PagePlaceholder'
import { useSession } from '../app/session/SessionContext'
import { LANGUAGE_CODES } from '../features/settings/model'

export function ChatPage() {
  const { settings, welcomeMessage } = useSession()
  return <PagePlaceholder icon="spark" eyebrow="YOUR RESEARCH WORKSPACE" title="Chat" description={welcomeMessage} descriptionLang={LANGUAGE_CODES[settings.language]} note="Your conversations and document tools will find a home here. Chat features are coming in a future update." />
}
