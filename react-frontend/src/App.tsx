import { BrowserRouter, Navigate, Route, Routes } from 'react-router'
import { AppLayout } from './components/layout/AppLayout'
import { AboutChatbotPage } from './pages/AboutChatbotPage'
import { AboutDeveloperPage } from './pages/AboutDeveloperPage'
import { ChatPage } from './pages/ChatPage'
import { SessionProvider } from './app/session/SessionProvider'

function App() {
  return (
    <SessionProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route index element={<ChatPage />} />
            <Route path="about-chatbot" element={<AboutChatbotPage />} />
            <Route path="about-developer" element={<AboutDeveloperPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </SessionProvider>
  )
}

export default App
