import { useEffect, useState } from 'react'
import './OpeningSplash.css'

const SPLASH_COMPLETION_KEY = 'researchmate.splash.completed.v1'
const REDUCED_MOTION = '(prefers-reduced-motion: reduce)'

function shouldShowSplash(): boolean {
  try {
    return !window.matchMedia(REDUCED_MOTION).matches
      && window.sessionStorage.getItem(SPLASH_COMPLETION_KEY) !== 'true'
  } catch {
    // If browser APIs are unavailable, go straight to the application.
    return false
  }
}

export function OpeningSplash({ onComplete }: { onComplete: () => void }) {
  const [visible, setVisible] = useState(shouldShowSplash)

  useEffect(() => {
    try {
      // Claim this tab immediately so even a refresh during the reveal skips it.
      window.sessionStorage.setItem(SPLASH_COMPLETION_KEY, 'true')
    } catch {
      // Storage restrictions must never prevent access to the application.
    }

    if (!visible) {
      onComplete()
      return
    }

    const dismiss = () => setVisible(false)
    const motionPreference = window.matchMedia(REDUCED_MOTION)
    const onMotionChange = () => {
      if (motionPreference.matches) dismiss()
    }
    motionPreference.addEventListener('change', onMotionChange)
    // Only a safety net: normal dismissal comes from the CSS animation ending.
    const fallback = window.setTimeout(dismiss, 3250)

    return () => {
      window.clearTimeout(fallback)
      motionPreference.removeEventListener('change', onMotionChange)
    }
  }, [visible, onComplete])

  if (!visible) return null

  return (
    <div
      className="opening-splash"
      aria-hidden="true"
      onAnimationEnd={(event) => {
        if (event.target === event.currentTarget && event.animationName === 'splash-reveal') {
          setVisible(false)
        }
      }}
    >
      <div className="opening-splash__brand">
        <div className="opening-splash__mark">
          <span className="opening-splash__glow" />
          <span className="opening-splash__ripple" />
          <span className="opening-splash__ripple opening-splash__ripple--second" />
          <span className="opening-splash__ripple opening-splash__ripple--third" />
          <img className="opening-splash__logo" src={`${import.meta.env.BASE_URL}researchmate.svg`} alt="" width="88" height="88" />
        </div>
        <p className="opening-splash__name">ResearchMate</p>
      </div>
    </div>
  )
}
