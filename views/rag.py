# Importing modules required for the chatbot functionality, including model setup, history management, and Streamlit UI
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers.string import StrOutputParser
from dotenv import load_dotenv

# Setting Up Langchain Tracing
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

# Defining a dictionary to map model names to their identifiers for API calls
model_dict = {
    "DeepSeek r1":"deepseek-r1-distill-qwen-32b",
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


# Initializing session state variables for chat history and user-assistant messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()  # Manages message history within the session

if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to retrieve session-specific chat history for maintaining conversation context
def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    return st.session_state.chat_history  # Returns the chat history for the current session


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a research-focused assistant. Provide detailed, evidence-based responses and reference credible sources when possible."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a research assistant. Utilize the provided context to deliver accurate, well-researched, and evidence-backed responses. Ensure responses are aligned with academic and research standards."),
        ("human", "Context: {context}"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# Displaying chat history to provide a consistent user experience
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Capturing user input from the chat input box
user_input = st.chat_input("Ask a question:")
if user_input:
    # Storing the user's input in session state and displaying it in the chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    if st.session_state.retriever:
        context = st.session_state.retriever.get_relevant_documents(user_input)
        chain=create_stuff_documents_chain(llm=model, prompt=rag_prompt)
        with_message_history = RunnableWithMessageHistory(
            chain,
            get_chat_history,  # Function to fetch chat history
            input_messages_key="messages",  # Key to access the messages
        )
    # Generating the assistant's response based on the chat history and input
        response = with_message_history.invoke(
            {"context":context,
             "language": st.session_state.language,
            "messages": [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]},
            config={"configurable": {"session_id": "default_rag_session"}},
        )
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

    else:
        chain=prompt | model
        with_message_history = RunnableWithMessageHistory(
            chain,
            get_chat_history,  # Function to fetch chat history
            input_messages_key="messages",  # Key to access the messages
        )
        response = with_message_history.invoke(
            {"language": st.session_state.language,
            "messages": [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]},
            config={"configurable": {"session_id": "default_session"}},
        )
        st.session_state.messages.append({"role": "assistant", "content": response.content})
        with st.chat_message("assistant"):
            st.write(response.content)

else:
    # Adding a welcome message at the start of the session
    with st.chat_message(""): 
        st.write(greetings[st.session_state.language])