# Importing modules required for the chatbot functionality, including model setup, history management, and Streamlit UI
import os
from dotenv import load_dotenv
import streamlit as st

# Import backend services and core logic
from services.llm import create_llm
from core.rag_pipeline import process_uploaded_pdf, answer_with_rag, answer_without_rag
from core.chat_history import StreamlitSessionChatHistory

load_dotenv()

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
model = create_llm(
    model_name=selected_model,
    temperature=temperature,
    max_tokens=max_tokens
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
            # Process PDF and create retriever using core RAG pipeline
            st.session_state.retriever = process_uploaded_pdf(uploaded_file)
            st.success("Document processed successfully!")


# Initialize chat history store
chat_history = StreamlitSessionChatHistory(session_key="rag_messages")

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
        # RAG mode: retrieve context and generate response using RAG pipeline
        response = answer_with_rag(
            llm=model,
            retriever=st.session_state.retriever,
            user_input=user_input,
            chat_history_langchain=chat_history.get_langchain_messages(),
            language=st.session_state.language
        )
        
        # Store and display assistant's response
        chat_history.add_message("assistant", response)
        with st.chat_message("assistant"):
            st.write(response)

    else:
        # Normal chat mode: generate response without RAG context
        response = answer_without_rag(
            llm=model,
            chat_history_langchain=chat_history.get_langchain_messages(),
            language=st.session_state.language
        )
        
        # Store and display assistant's response
        chat_history.add_message("assistant", response)
        with st.chat_message("assistant"):
            st.write(response)

else:
    # Adding a welcome message at the start of the session
    with st.chat_message(""): 
        st.write(greetings[st.session_state.language])