# Importing modules required for the chatbot functionality, including model setup, history management, and Streamlit UI
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
import os
import tempfile
from dotenv import load_dotenv
load_dotenv()


os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')

# Setting Up Langchain Tracing
os.environ['LANGCHAIN_TRACING_V2']="true"

# Defining a dictionary to map model names to their identifiers for API calls
model_dict = {
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
            ["LLaMA 3.1-8B", "Gemma2 9B", "Mixtral"],
            help="Select the model type you want to use for generating responses. Each model has different strengths and use cases.",
        )
        model_desc = {
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
            256,
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
st.session_state.model = ChatGroq(
    model=selected_model,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=temperature,
    max_tokens=max_tokens,
    streaming=True,
)

# Setting up the main Streamlit interface and initializing the chatbot UI
st.title("IntelliChat 🤖")
st.write("Your intelligent assistant developed by Mitesh😎, ready to answer your queries!")


# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "model" not in st.session_state:
    st.session_state.model = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key="your_groq_api_key")

# Function to retrieve session-specific chat history
def get_chat_history(session_id: str):
    return st.session_state.chat_history

# Generic template without retrieval
generic_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant developed by Mitesh. Answer all questions to the best of your ability and give response in given {language} language."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# RAG-enabled template
rag_generic_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant developed by Mitesh. Use the provided context to answer accurately. Give responses in {language} language."),
        ("human", "Context: {context}"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

with st.sidebar:
    with st.expander("**Upload your Research Paper**", icon="📄"):
        upload = st.file_uploader("", type=["pdf", "docx"])

# Ensure retriever is built only once per uploaded file
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
    st.session_state.retriever = None

def build_retriever(upload):
    """Loads document, splits text, and builds FAISS retriever."""
    if upload is None:
        return None

    # Create temporary file to store uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + upload.name.split(".")[-1]) as temp_file:
        temp_file.write(upload.read())
        temp_path = temp_file.name

    # Select correct loader
    if upload.type == "application/pdf":
        loader = PyMuPDFLoader(temp_path)
    elif upload.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        loader = Docx2txtLoader(temp_path)
    else:
        st.error("Unsupported file type.")
        return None

    # Load and process document
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Initialize embeddings and FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    
    return db.as_retriever()

if upload and upload.name != st.session_state.processed_file:
    st.session_state.processed_file = upload.name  # Store file name to track changes
    with st.sidebar.status("Processing document...", state="running"):
        st.session_state.retriever = build_retriever(upload)

# Define the RAG pipeline based on retriever availability
if st.session_state.retriever:
    chain = st.session_state.retriever | rag_generic_template | st.session_state.model
else:
    chain = generic_template | st.session_state.model

# Initialize chat history if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Ensure session-specific chat history exists
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Function to retrieve session-specific chat history
def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    return st.session_state.chat_history

# Create message history-aware RAG pipeline
with_message_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="messages",
)

# Display past messages (persists across reruns)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Capture user input
user_input = st.chat_input("Ask a question:")

if user_input:
    # Append user message to session history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Prepare query data
    retrieved_docs = st.session_state.retriever.invoke(user_input) if upload else None

    query_data = {
        "language": st.session_state.language,
        "context": "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No additional context available.",
        "messages": [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages],
    }


    # Generate assistant response
    response = with_message_history.invoke(
        query_data, 
        config={"configurable": {"session_id": "default_session"}}
    )

    # Append assistant response to session history
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    with st.chat_message("assistant"):
        st.write(response.content)

else:
    # Show welcome message only when no user input is detected
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.write("👋 Hello! Upload a research paper or ask a question!")
