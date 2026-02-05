# Unified Chat View
# Combines normal chat, RAG, and Agent workflows in a single Streamlit UI

import html
import uuid
import streamlit as st
from frontend.css import render_message, render_mode_banner, render_processing_banner, render_generating_banner, render_success_banner, render_error_banner
# Import API client for backend communication
from frontend.api_client import (
    send_chat_message,
    send_rag_query,
    send_agent_message,
    upload_document,
    get_chat_history,
    clear_chat_history,
    APIError
)

# Scoped CSS for layout customization
st.markdown(
    """
    <style>
    :root {
        --chat-max-width: 750px;
    }
    .block-container {
        padding-left: 10rem;
        padding-right: 10rem;
        max-width: 1400px;
    }
    .content-wrapper {
        margin: 0 auto;
    }
    .chat-wrapper {
        max-width: var(--chat-max-width);
        margin: 0 auto;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    section[data-testid="stChatInput"] {
        max-width: var(--chat-max-width);
        margin-left: auto;
        margin-right: auto;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Generate a unique session ID for this user session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize document_id in session state
if "document_id" not in st.session_state:
    st.session_state.document_id = None

# Initialize agent mode in session state
if "agent_enabled" not in st.session_state:
    st.session_state.agent_enabled = False


# Setting up the sidebar for user customization options
with st.sidebar:
    # Agent Mode Toggle
    with st.expander("**Agent Mode**", icon="🤖"):
        agent_enabled = st.toggle(
            "Enable Agent Mode",
            value=st.session_state.agent_enabled,
            help="When enabled, your queries will be processed by the AI agent which can perform complex research tasks."
        )
        st.session_state.agent_enabled = agent_enabled
        
        if agent_enabled:
            st.info("Agent Mode is ON. Your queries will be routed to the agent for advanced processing.")
        else:
            st.caption("Agent Mode is OFF. Using normal chat or RAG based on document upload.")

    # File uploader for PDF documents (RAG functionality)
    with st.expander("**Attach a Document**", icon="📎"):
        uploaded_file=st.file_uploader("Upload a pdf",type=["pdf"], help="Upload a PDF document to enable Retrieval-Augmented Generation (RAG) mode. The AI will use the content of the uploaded document to answer your queries more accurately.")

    # Adding a dropdown for language selection to support multilingual capabilities
    with st.expander("**Language Options**", icon="🌐"):
        language = st.selectbox(
            "Select Model Language",
            ["English", "Hindi", "Spanish", "French", "German"],
        )
        st.session_state.language = language

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
        st.session_state.model = model_type
        st.markdown(f"**Selected Model:** {model_type}", help=model_desc[model_type])

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

# Setting up the main Streamlit interface
st.markdown(
    "<h1 style='text-align: center;'>Research Mate 🤖</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Your research-oriented assistant developed by Mitesh😎, ready to assist with academic and research queries!</p>",
    unsafe_allow_html=True
)

# Mode indicator banner (replace st.caption usage)

# Call this where you previously used st.caption(...)
render_mode_banner()

# Process the uploaded document via API (Only when a new file is uploaded)
if uploaded_file:
    if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        
        status_placeholder = st.empty()
        render_processing_banner(status_placeholder, "Processing document…")
        
        try:
            result = upload_document(uploaded_file)
            st.session_state.document_id = result["document_id"]
            status_placeholder.empty()
            
            success_placeholder = st.empty()
            render_success_banner(success_placeholder, "Document processed successfully!")
        except APIError as e:
            st.session_state.document_id = None
            status_placeholder.empty()
            
            error_placeholder = st.empty()
            render_error_banner(error_placeholder, f"Failed to upload document: {e.message}")


# Fetch and display chat history from backend
def display_chat_history():
    """Fetch chat history from backend and display it."""
    try:
        messages = get_chat_history(st.session_state.session_id)
        for msg in messages:
            render_message(msg["role"], msg["content"])
        return messages
    except APIError:
        # If history fetch fails, just show empty chat
        return []


# Display existing chat history
existing_messages = display_chat_history()

# Capturing user input from the chat input box
user_input = st.chat_input("Ask a question...")

if user_input:
    # Display user's message immediately
    render_message("user", user_input)

    # Reuse the same styled status banners (no spinner)
    status_placeholder = st.empty()

    try:
        render_generating_banner(status_placeholder, "Generating response…")

        if st.session_state.agent_enabled:
            # Agent mode: send query to agent endpoint
            result = send_agent_message(
                session_id=st.session_state.session_id,
                message=user_input,
                document_id=st.session_state.document_id,
                language=st.session_state.language,
                model_type=st.session_state.get("model", None),
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif st.session_state.document_id:
            # RAG mode: send query to RAG endpoint
            result = send_rag_query(
                session_id=st.session_state.session_id,
                document_id=st.session_state.document_id,
                message=user_input,
                language=st.session_state.language,
                model_type=st.session_state.get("model", None),
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            # Normal chat mode: send message to chat endpoint
            result = send_chat_message(
                session_id=st.session_state.session_id,
                message=user_input,
                language=st.session_state.language,
                model_type=st.session_state.get("model", None),
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        status_placeholder.empty()
        
        # Display assistant's response
        render_message("assistant", result.get("content", ""))
                
    except APIError as e:
        status_placeholder.empty()
        
        error_placeholder = st.empty()
        render_error_banner(error_placeholder, f"Failed to generate response: {e.message}")

else:
    # Adding a welcome message at the start of the session (only if no history)
    if not existing_messages:
        render_message("assistant", greetings[st.session_state.language])
