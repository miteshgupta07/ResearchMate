import { Link } from 'react-router'
import { AboutCard, AboutSection } from '../components/about/AboutSection'
import { Icon } from '../components/ui/Icon'

const purposes = [
  ['Document-grounded answers', 'Ask questions about uploaded research papers using relevant passages retrieved from the document.'],
  ['Research productivity', 'Find contextually relevant information and explore academic papers with less manual searching.'],
  ['Accuracy, grounding & transparency', 'Use retrieved document context to reduce speculation. Compare answers with the original source; structured citations are not currently returned by the RAG API.'],
  ['User-controlled agents', 'Choose when to enable Agent Mode for document questions, summaries, and real-time arXiv search and recommendations.'],
]

const technologies = [
  ['Backend · FastAPI', 'HTTP APIs orchestrate conversations, RAG pipelines, PDF ingestion, and session history. The frontend and backend run independently. PDF processing completes before its upload request returns.'],
  ['Frontends · React & Streamlit', 'The React workspace and original Streamlit application coexist. Both provide chat, session management, PDF attachment, and runtime generation settings through the backend.'],
  ['Data & storage · PostgreSQL & FAISS', 'PostgreSQL persists session-based chat history. FAISS indexes document embeddings for vector similarity search and retrieval.'],
  ['Language models · Groq & Hugging Face', 'Groq provides language-model integration, with Hugging Face embeddings used for retrieval. GPT OSS is the current selectable model; temperature and token limits can be adjusted at runtime.'],
]

const features = [
  ['Persistent document ingestion', 'Upload, chunk, embed, and index research papers on the backend for document-based querying. Indexes persist on the server; the selected attachment in React is kept in memory and is not restored after a reload.'],
  ['Evaluated RAG pipeline', 'A separate evaluation framework measures retrieval quality using metrics such as Recall@K. Evaluation supports pipeline improvement without implying that every live response has been scored.'],
  ['Agent Mode toggle', 'Manually enable intelligent routing to normal chat, document Q&A, document summarization, arXiv search, recommendations, or fallback.'],
  ['Dynamic LLM configuration', 'Choose GPT OSS, adjust temperature, and control maximum tokens directly in the interface. Five response languages are available: English, Hindi, Spanish, French, and German.'],
  ['Auto-continuation', 'The backend can continue responses that hit generation limits, helping produce more complete answers. No manual continuation or streaming control is required.'],
  ['Chat history persistence', 'Conversation messages saved by the backend remain in PostgreSQL and can be retrieved with the same session ID. Agent paper-search and recommendation results are not currently saved by the backend.'],
]

const benefits = [
  ['Reliable research assistance', 'Grounding answers in uploaded documents helps reduce speculation and keeps research context close at hand.'],
  ['Transparent answers', 'Retrieved source material supports answer grounding. Verify findings against the original paper; the current interface does not supply structured source citations.'],
  ['Scalable architecture', 'The decoupled client-server design allows the frontend and backend to scale and evolve independently.'],
  ['Extensible design', 'Modular components and services support future integration of models, storage backends, and retrieval methods.'],
]

export function AboutChatbotPage() {
  return (
    <article className="mx-auto w-full max-w-5xl px-6 py-10 sm:px-10 sm:py-14">
      <header className="border-b border-line pb-10">
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-accent/20 bg-accent/5 px-3 py-1.5 text-xs text-accent"><Icon name="spark" className="size-4" /> GET TO KNOW RESEARCHMATE</div>
        <h1 className="text-3xl font-semibold tracking-tight sm:text-5xl">About ResearchMate<span className="text-accent">.</span></h1>
        <p className="mt-5 max-w-2xl text-lg leading-8 text-ink/80">A research assistant built around your questions, your papers, and the context that connects them.</p>
        <Link to="/" className="mt-7 inline-flex min-h-11 items-center gap-2 rounded-xl bg-accent px-5 text-sm font-semibold text-sidebar hover:opacity-90"><Icon name="chat" className="size-4" /> Open your workspace</Link>
      </header>
      <AboutSection id="chatbot-introduction" title="Introduction" icon="info">
        <div className="max-w-3xl space-y-4 text-sm leading-8 text-muted">
          <p>Developed by <Link to="/about-developer" className="rounded-sm text-ink underline decoration-accent/50 underline-offset-4 hover:text-accent">Mitesh Gupta</Link>, ResearchMate is an AI research assistant built on Retrieval-Augmented Generation (RAG) principles. It combines a FastAPI backend with a React workspace and the original Streamlit frontend to support academic and research queries with document-grounded answers.</p>
          <p>The system is designed with scalability and reliability in mind, featuring persistent storage, evaluated retrieval pipelines, and flexible language-model integration for serious research workflows.</p>
        </div>
      </AboutSection>
      <AboutSection id="chatbot-purpose" title="Purpose" icon="spark">
        <div className="grid gap-4 sm:grid-cols-2">{purposes.map(([title, text]) => <AboutCard key={title} title={title}><p>{text}</p></AboutCard>)}</div>
      </AboutSection>
      <AboutSection id="chatbot-architecture" title="Architecture & technologies" icon="panel">
        <p className="mb-5 text-sm leading-7 text-muted">A decoupled client-server architecture keeps the research interface, AI services, and data storage independently maintainable.</p>
        <div className="grid gap-4 sm:grid-cols-2">{technologies.map(([title, text]) => <AboutCard key={title} title={title}><p>{text}</p></AboutCard>)}</div>
      </AboutSection>
      <AboutSection id="chatbot-features" title="Key features" icon="settings">
        <div className="grid gap-4 sm:grid-cols-2">{features.map(([title, text]) => <AboutCard key={title} title={title}><p>{text}</p></AboutCard>)}</div>
      </AboutSection>
      <AboutSection id="chatbot-evaluation" title="Evaluation & reliability" icon="info">
        <div className="rounded-2xl border border-accent/20 bg-accent/5 p-5 sm:p-7">
          <p className="text-sm leading-8 text-ink/80">ResearchMate incorporates evaluation practices to assess retrieval and response quality using open benchmarks and standard retrieval metrics. Retrieval accuracy and answer grounding help assess whether responses are backed by source documents, with a focus on reducing hallucinations and improving trust.</p>
          <p className="mt-4 text-sm leading-8 text-muted">This evaluation-driven approach is intended to support dependable research assistance. It does not guarantee that every generated answer is correct; important findings should be checked against the source material.</p>
        </div>
      </AboutSection>
      <AboutSection id="chatbot-benefits" title="Benefits" icon="spark">
        <div className="grid gap-4 sm:grid-cols-2">{benefits.map(([title, text]) => <AboutCard key={title} title={title}><p>{text}</p></AboutCard>)}</div>
      </AboutSection>
      <p className="mt-12 border-t border-line pt-7 text-sm leading-7 text-muted">Built to support serious research workflows with accuracy, transparency, and scalability at its core.</p>
    </article>
  )
}
