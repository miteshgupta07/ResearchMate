# Importing modules required for the chatbot functionality
import uuid
import streamlit as st

# Import API client for backend communication
from services.api_client import (
    send_chat_message,
    send_rag_query,
    upload_document,
    get_chat_history,
    clear_chat_history,
    APIError
)

# Generate a unique session ID for this user session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

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

# Setting up the main Streamlit interface and initializing the chatbot UI
st.title("Research Mate 🤖")
st.write("Your research-oriented assistant developed by Mitesh😎, ready to assist with academic and research queries!")

uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# Initialize session state for document_id (replaces local retriever)
if "document_id" not in st.session_state:
    st.session_state.document_id = None

# Process the uploaded document via API (Only when a new file is uploaded)
if uploaded_file:
    if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        with st.spinner("Processing document..."):
            try:
                # Upload PDF via API - backend handles all processing
                result = upload_document(uploaded_file)
                st.session_state.document_id = result["document_id"]
                st.success("Document processed successfully!")
            except APIError as e:
                st.error(f"Failed to upload document: {e.message}")
                st.session_state.document_id = None


# Fetch and display chat history from backend
def display_chat_history():
    """Fetch chat history from backend and display it."""
    try:
        messages = get_chat_history(st.session_state.session_id)
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
    except APIError:
        # If history fetch fails, just show empty chat
        pass


# Display existing chat history
display_chat_history()

# Capturing user input from the chat input box
user_input = st.chat_input("Ask a question:")
if user_input:
    # Display user's message immediately
    with st.chat_message("user"):
        st.write(user_input)

    try:
        if st.session_state.document_id:
            # RAG mode: send query to RAG endpoint
            with st.spinner("Thinking..."):
                result = send_rag_query(
                    session_id=st.session_state.session_id,
                    document_id=st.session_state.document_id,
                    message=user_input,
                    language=st.session_state.language
                )
            
            # Display assistant's response
            with st.chat_message("assistant"):
                st.write(result["content"])
        else:
            # Normal chat mode: send message to chat endpoint
            with st.spinner("Thinking..."):
                result = send_chat_message(
                    session_id=st.session_state.session_id,
                    message=user_input,
                    language=st.session_state.language
                )
            
            # Display assistant's response
            with st.chat_message("assistant"):
                st.write(result["content"])
                
    except APIError as e:
        with st.chat_message("assistant"):
            st.error(f"Error: {e.message}")

else:
    # Adding a welcome message at the start of the session (only if no history)
    try:
        messages = get_chat_history(st.session_state.session_id)
        if not messages:
            with st.chat_message("assistant"): 
                st.write(greetings[st.session_state.language])
    except APIError:
        with st.chat_message("assistant"): 
            st.write(greetings[st.session_state.language])