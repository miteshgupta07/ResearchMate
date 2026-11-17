import { AboutCard, AboutSection } from '../components/about/AboutSection'
import { Icon } from '../components/ui/Icon'

const skillGroups = [
  ['Data Science & AI', ['Python', 'Machine Learning', 'Deep Learning', 'CV', 'NLP', 'Generative AI', 'Feature Engineering', 'Model Optimization']],
  ['MLOps & Deployment', ['AWS', 'Docker', 'Kubernetes', 'MLflow', 'FastAPI', 'Model Versioning', 'CI/CD Pipelines', 'Hugging Face', 'LLM Evaluation']],
  ['LLM & Agent Frameworks', ['MCP', 'LangGraph', 'LangChain', 'TensorFlow', 'Keras', 'PyTorch', 'Scikit-learn', 'OpenCV', 'NLTK', 'spaCy', 'Streamlit']],
  ['Data Analysis & Visualization', ['Data Preprocessing', 'Data Visualization', 'Data Analysis', 'SQL', 'Pandas', 'NumPy']],
  ['Databases & Vector Stores', ['MySQL', 'PostgreSQL', 'Chroma', 'FAISS']],
  ['Version Control & Tools', ['Git', 'GitHub']],
] as const

const interests = [
  ['AI Research', 'Staying updated with state-of-the-art AI models and research papers.'],
  ['Projects', 'Working on innovative tools like production-grade chatbots and RAG systems.'],
  ['Community', 'Sharing knowledge through blogs and contributing to open-source projects.'],
]

const contactLinks = [
  ['Email', 'mailto:miteshgupta2711@gmail.com'],
  ['LinkedIn', 'https://www.linkedin.com/in/mitesh-gupta/'],
  ['X / Twitter', 'https://x.com/mg_mitesh'],
  ['Instagram', 'https://www.instagram.com/mg_mitesh_gupta/'],
  ['GitHub', 'https://github.com/miteshgupta07'],
]

export function AboutDeveloperPage() {
  const resumeUrl = `${import.meta.env?.BASE_URL ?? '/'}resume/${encodeURIComponent('Mitesh Gupta Resume-October 2026.pdf')}`
  return (
    <article className="mx-auto w-full max-w-5xl px-6 py-10 sm:px-10 sm:py-14">
      <header className="border-b border-line pb-10">
        <p className="mb-6 text-xs font-medium tracking-[0.18em] text-accent">ABOUT THE DEVELOPER</p>
        <div className="flex flex-col items-start gap-6 sm:flex-row sm:items-center">
          <div aria-hidden="true" className="flex size-24 shrink-0 items-center justify-center rounded-3xl border border-accent/20 bg-accent/10 text-3xl font-semibold tracking-tight text-accent">MG</div>
          <div>
            <h1 className="text-3xl font-semibold tracking-tight sm:text-5xl">Mitesh Gupta<span className="text-accent">.</span></h1>
            <p className="mt-3 text-lg text-muted">AI Engineer · Developer of ResearchMate</p>
          </div>
        </div>
        <div className="mt-8 flex flex-wrap gap-3">
          <a href="https://miteshgupta-portfolio.vercel.app/" target="_blank" rel="noopener noreferrer" className="inline-flex min-h-11 items-center gap-2 rounded-xl bg-accent px-5 text-sm font-semibold text-sidebar hover:opacity-90"><Icon name="panel" className="size-4" /> Visit my portfolio<span className="sr-only"> (opens in a new tab)</span></a>
          <a href={resumeUrl} download="Mitesh Gupta Resume.pdf" className="inline-flex min-h-11 items-center gap-2 rounded-xl border border-line bg-surface px-5 text-sm font-medium text-ink hover:border-accent/50"><Icon name="person" className="size-4" /> Download my resume <span className="text-xs text-muted">PDF</span></a>
        </div>
      </header>
      <AboutSection id="developer-introduction" title="Introduction" icon="person">
        <p className="max-w-3xl text-base leading-8 text-ink/80">Hi, I’m Mitesh Gupta, an AI Engineer with strong expertise in Machine Learning, Deep Learning, NLP, and Generative AI.</p>
      </AboutSection>
      <AboutSection id="developer-experience" title="Experience & approach" icon="panel">
        <div className="grid gap-4 sm:grid-cols-2">
          <AboutCard title="Building & deploying AI systems"><p>Experienced in building and deploying scalable AI solutions and backend systems using Python and FastAPI.</p></AboutCard>
          <AboutCard title="From problem to production"><p>Skilled in end-to-end model development, data processing, API design, and cloud deployment. Adept at translating complex business problems into practical, production-ready AI applications.</p></AboutCard>
        </div>
      </AboutSection>
      <AboutSection id="developer-skills" title="Skills" icon="settings">
        <div className="grid gap-4 sm:grid-cols-2">
          {skillGroups.map(([title, skills]) => (
            <AboutCard key={title} title={title}><ul className="flex flex-wrap gap-2">{skills.map((skill) => <li key={skill} className="rounded-lg border border-line bg-canvas/60 px-2.5 py-1 text-xs leading-6 text-ink/80">{skill}</li>)}</ul></AboutCard>
          ))}
        </div>
      </AboutSection>
      <AboutSection id="developer-interests" title="Interests" icon="spark">
        <div className="grid gap-4 lg:grid-cols-3">{interests.map(([title, text]) => <AboutCard key={title} title={title}><p>{text}</p></AboutCard>)}</div>
      </AboutSection>
      <AboutSection id="developer-contact" title="Contact me" icon="chat">
        <div className="rounded-2xl border border-accent/20 bg-accent/5 p-5 sm:p-7">
          <p className="text-lg font-medium text-ink">Let’s build something amazing together.</p>
          <p className="mt-2 text-sm leading-7 text-muted">Thank you for visiting my chatbot application. Connect with me through any of the channels below.</p>
          <ul className="mt-6 flex flex-wrap gap-3">
            {contactLinks.map(([label, href]) => (
              <li key={label}><a href={href} target={href.startsWith('https:') ? '_blank' : undefined} rel={href.startsWith('https:') ? 'noopener noreferrer' : undefined} className="inline-flex min-h-11 items-center gap-2 rounded-xl border border-line bg-canvas px-4 text-sm text-ink transition-colors hover:border-accent/50 hover:text-accent">{label}<span aria-hidden="true">↗</span>{href.startsWith('https:') && <span className="sr-only"> (opens in a new tab)</span>}</a></li>
            ))}
          </ul>
        </div>
      </AboutSection>
    </article>
  )
}
