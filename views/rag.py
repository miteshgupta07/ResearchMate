# Importing modules required for the chatbot functionality, including model setup, history management, and Streamlit UI
import os

from dotenv import load_dotenv

import streamlit as st

from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# Custom chat memory abstraction to replace LangChain memory management
class ChatMessage:
    """Represents a single chat message with role and content."""
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
    
    def to_dict(self):
        """Convert to dictionary format for storage and display."""
        return {"role": self.role, "content": self.content}
    
    def to_langchain_tuple(self):
        """Convert to LangChain message format (role, content) tuple."""
        if self.role == "user":
            return ("human", self.content)
        elif self.role == "assistant":
            return ("ai", self.content)
        else:
            return (self.role, self.content)


class ChatHistoryStore:
    """Manages chat history backed by Streamlit session state."""
    def __init__(self, session_key: str = "rag_messages"):
        self.session_key = session_key
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = []
    
    def add_message(self, role: str, content: str):
        """Add a new message to the chat history."""
        message = ChatMessage(role, content)
        st.session_state[self.session_key].append(message.to_dict())
    
    def get_messages(self):
        """Get all messages in dictionary format."""
        return st.session_state[self.session_key]
    
    def get_langchain_messages(self):
        """Convert messages to LangChain format for prompt templates."""
        return [ChatMessage(**msg).to_langchain_tuple() for msg in st.session_state[self.session_key]]
    
    def clear(self):
        """Clear all messages from history."""
        st.session_state[self.session_key] = []

# Setting Up Langchain Tracing
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

# Defining a dictionary to map model names to their identifiers for API calls
model_dict = {
    "DeepSeek r1":"llama-3.1-8b-instant",
    "LLaMA 3.1-8B": "llama-3.1-8b-instant",
    "Gemma2 9B": "gemma2-9b-it",
    "Mixtral": "mixtral-8x7b-32768",
}

# Setting up the sidebar for user customization options
with st.sidebar:
    # Adding a dropdown for language selection to support multilingual capabilities
    with st.expander("**Language Options**",icon="🌐"):
        language = st.selectbox(
            "Select Model Language",
            ["English", "Hindi", "Spanish", "French", "German"],
        )
        st.session_state.language = language  # Storing the selected language in session state

    # Adding an expandable section for model customization
    with st.expander("**Model Customization**", icon="🛠️"):
        # Allowing the user to select the model type for generating responses
        model_type = st.selectbox(
            "**Choose model type**",
            ["DeepSeek r1","LLaMA 3.1-8B", "Gemma2 9B", "Mixtral"],
            help="Select the model type you want to use for generating responses. Each model has different strengths and use cases.",
        )
        model_desc = {
            "DeepSeek r1":"DeepSeek's initial large language model, known for its robust research-oriented capabilities and strong performance in coding and multilingual reasoning tasks.",
            "LLaMA 3.1-8B": "LLaMA (Large Language Model Meta AI) 3.1-8B is a versatile language model developed by Meta, featuring 8 billion parameters. It excels in a variety of natural language processing tasks such as text generation, summarization, and translation, while maintaining efficiency and reliability in performance.",
            "Gemma2 9B": "Gemma2 is a large-scale language model with 9 billion parameters, known for its ability to generate highly coherent, contextually accurate, and nuanced text. It is suited for applications that require creative content generation, such as dialogue systems, storytelling, and more.",
            "Mixtral": "Mixtral is a multi-modal AI model optimized for both text and image processing. This model integrates visual and textual information to enable tasks like image captioning, text-to-image generation, and interactive storytelling, offering a creative approach to AI applications."
            }
        
        # Displaying detailed descriptions for each model based on user selection
        st.session_state.model=model_type
        st.markdown(f"**Selected Model:** {model_type}",help=model_desc[model_type])

        # Adding sliders to allow fine-tuning of model parameters
        temperature = st.slider(
            "**Temperature**",
            0.0,
            1.0,
            0.7,
            help="Controls the creativity of the model's responses. Higher values (closer to 1.0) produce more creative and diverse outputs, while lower values (closer to 0.0) result in more focused and deterministic responses.",
        )
        max_tokens = st.slider(
            "**Max Tokens**",
            1,
            2048,
            512,
            help="Controls the maximum number of tokens the model can generate in its response. Higher values allow for longer responses.",
        )

# Displaying a greeting message based on the selected language
greetings = {
    "English": "Hi! How can I assist you today?",
    "Spanish": "¡Hola! ¿Cómo puedo ayudarte hoy?",
    "French": "Bonjour! Comment puis-je vous aider aujourd'hui?",
    "German": "Hallo! Wie kann ich Ihnen heute helfen?",
    "Hindi": "नमस्ते! मैं आज आपकी कैसे मदद कर सकता हूँ?",
}

# Selecting the appropriate model identifier for API calls based on the user's choice
selected_model = model_dict[st.session_state.model]

# Initializing the language model with parameters and enabling streaming for real-time responses
model = ChatGroq(
    model=selected_model,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=temperature,
    max_tokens=max_tokens,
    streaming=True,
    verbose=False
)

# Setting up the main Streamlit interface and initializing the chatbot UI
st.title("Research Mate 🤖")
st.write("Your research-oriented assistant developed by Mitesh😎, ready to assist with academic and research queries!")

uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# Initialize session state for retriever
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Process the uploaded document (Only when a new file is uploaded)
if uploaded_file:
    if "uploaded_file" not in st.session_state or st.session_state.uploaded_file != uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        with st.spinner("Processing document..."):
            # Save the uploaded file temporarily
            with open("uploaded_document.pdf", "wb") as f:
                f.write(st.session_state.uploaded_file.read())

            # Load the document
            loader = PyMuPDFLoader("uploaded_document.pdf")
            documents = loader.load()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = text_splitter.split_documents(documents)

            # Embeddings and FAISS Vector Store
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(docs, embeddings)
            st.session_state.retriever = db.as_retriever()

            st.success("Document processed successfully!")


# Initialize chat history store
chat_history = ChatHistoryStore()


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a research-focused assistant. Provide detailed, evidence-based responses and reference credible sources when possible."),
        MessagesPlaceholder(variable_name="rag_messages"),
    ]
)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a research assistant. Utilize the provided context to deliver accurate, well-researched, and evidence-backed responses. Ensure responses are aligned with academic and research standards."),
        ("human", "Context: {context}"),
        MessagesPlaceholder(variable_name="rag_messages"),
    ]
)


# Displaying chat history to provide a consistent user experience
for msg in chat_history.get_messages():
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Capturing user input from the chat input box
user_input = st.chat_input("Ask a question:")
if user_input:
    # Storing the user's input in chat history and displaying it
    chat_history.add_message("user", user_input)
    with st.chat_message("user"):
        st.write(user_input)

    if st.session_state.retriever:
        # RAG mode: retrieve context and generate response
        context = st.session_state.retriever.invoke(user_input)
        chain = create_stuff_documents_chain(llm=model, prompt=rag_prompt)
        
        # Generate response with context and message history
        response = chain.invoke({
            "context": context,
            "language": st.session_state.language,
            "rag_messages": chat_history.get_langchain_messages()
        })
        
        # Store and display assistant's response
        chat_history.add_message("assistant", response)
        with st.chat_message("assistant"):
            st.write(response)

    else:
        # Normal chat mode: generate response without RAG context
        chain = prompt | model
        
        # Generate response with message history
        response = chain.invoke({
            "language": st.session_state.language,
            "rag_messages": chat_history.get_langchain_messages()
        })
        
        # Store and display assistant's response
        chat_history.add_message("assistant", response.content)
        with st.chat_message("assistant"):
            st.write(response.content)

else:
    # Adding a welcome message at the start of the session
    with st.chat_message(""): 
        st.write(greetings[st.session_state.language])