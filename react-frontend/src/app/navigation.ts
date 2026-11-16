import type { IconName } from '../components/ui/Icon'

export const navigation: { path: string; label: string; icon: IconName }[] = [
  { path: '/', label: 'Chat', icon: 'chat' },
  { path: '/about-chatbot', label: 'About Chatbot', icon: 'info' },
  { path: '/about-developer', label: 'About Developer', icon: 'person' },
]
