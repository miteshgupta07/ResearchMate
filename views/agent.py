# Importing necessary modules
import uuid
import streamlit as st

# Import API client for backend communication
from services.api_client import (
    send_chat_message,
    get_chat_history,
    clear_chat_history,
    APIError
)

# Generate a unique session ID for this user session (separate from RAG session)
if "agent_session_id" not in st.session_state:
    st.session_state.agent_session_id = str(uuid.uuid4())

# Sidebar for customization options
with st.sidebar:
    with st.expander("**Language Options**", icon="🌐"):
        language = st.selectbox(
            "Select Model Language",
            ["English", "Hindi", "Spanish", "French", "German"],
        )
        st.session_state["language"] = language

    with st.expander("**Model Customization**", icon="🛠️"):
        model_type = st.selectbox(
            "**Choose model type**",
            ["DeepSeek","LLaMA 3.1-8B", "Gemma2 9B", "Mixtral"],
        )
        st.session_state["model"] = model_type

        temperature = st.slider(
            "**Temperature**",
            0.0,
            1.0,
            0.7,
        )
        max_tokens = st.slider(
            "**Max Tokens**",
            1,
            2048,
            512,
        )

# Greetings based on language selection
greetings = {
    "English": "Hi! How can I assist you today?",
    "Spanish": "¡Hola! ¿Cómo puedo ayudarte hoy?",
    "French": "Bonjour! Comment puis-je vous aider aujourd'hui?",
    "German": "Hallo! Wie kann ich Ihnen heute helfen?",
    "Hindi": "नमस्ते! मैं आज आपकी कैसे मदद कर सकता हूँ?",
}

# Main interface
st.title("Research Mate 🤖")
st.write("Your research-oriented assistant developed by Mitesh😎, ready to assist with academic and research queries!")


# Fetch and display chat history from backend
def display_agent_chat_history():
    """Fetch chat history from backend and display it."""
    try:
        messages = get_chat_history(st.session_state.agent_session_id)
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
    except APIError:
        # If history fetch fails, just show empty chat
        pass


# Display existing chat history
display_agent_chat_history()

# User input
user_input = st.chat_input("Ask a question...")

if user_input:
    # Display user's message immediately
    with st.chat_message("user"):
        st.write(user_input)

    try:
        # Send message to chat endpoint
        with st.spinner("Thinking..."):
            result = send_chat_message(
                session_id=st.session_state.agent_session_id,
                message=user_input,
                language=st.session_state["language"]
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
        messages = get_chat_history(st.session_state.agent_session_id)
        if not messages:
            with st.chat_message("assistant"): 
                st.write(greetings[st.session_state["language"]])
    except APIError:
        with st.chat_message("assistant"): 
            st.write(greetings[st.session_state["language"]])
