# Importing necessary modules
import uuid
import requests
import streamlit as st

# Backend API configuration
API_BASE_URL = "http://localhost:8000"

# Generate a unique session ID for this user session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize message history in session state
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

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
            ["DeepSeek r1","LLaMA 3.1-8B", "Gemma2 9B", "Mixtral"],
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


def send_agent_message(
    session_id: str,
    message: str,
    document_id: str = None,
    language: str = None,
    model_type: str = None,
    temperature: float = None,
    max_tokens: int = None
) -> dict:
    """Send a message to the agent backend and return the response."""
    payload = {
        "session_id": session_id,
        "message": message,
        "document_id": document_id,
        "language": language,
        "model_type": model_type,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    response = requests.post(
        f"{API_BASE_URL}/agent/route",
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    return response.json()


# Display existing chat history from session state
for msg in st.session_state.agent_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Show welcome message if no messages yet
if not st.session_state.agent_messages:
    with st.chat_message("assistant"):
        st.write(greetings[st.session_state.get("language", "English")])

# User input
user_input = st.chat_input("Ask a question...")

if user_input:
    # Add user message to history
    st.session_state.agent_messages.append({"role": "user", "content": user_input})
    
    # Display user's message immediately
    with st.chat_message("user"):
        st.write(user_input)

    try:
        # Send message to agent endpoint
        with st.spinner("Thinking..."):
            # Get document_id if available in session state
            document_id = st.session_state.get("document_id", None)
            
            result = send_agent_message(
                session_id=st.session_state.session_id,
                message=user_input,
                document_id=document_id,
                language=st.session_state.get("language", "English"),
                model_type=st.session_state.get("model", None),
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        # Add assistant response to history
        assistant_content = result.get("content", "")
        st.session_state.agent_messages.append({"role": "assistant", "content": assistant_content})
        
        # Display assistant's response
        with st.chat_message("assistant"):
            st.write(assistant_content)
            
    except requests.exceptions.RequestException as e:
        error_message = f"Error communicating with backend: {str(e)}"
        st.session_state.agent_messages.append({"role": "assistant", "content": error_message})
        with st.chat_message("assistant"):
            st.error(error_message)
