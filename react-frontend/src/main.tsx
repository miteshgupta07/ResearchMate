import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { OpeningSplash } from './components/ui/OpeningSplash'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <OpeningSplash />
    <App />
  </StrictMode>,
)
