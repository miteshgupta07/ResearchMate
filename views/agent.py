# Importing necessary modules
import streamlit as st
import os
from dotenv import load_dotenv

# LangChain & AI Model Imports
from langchain_groq import ChatGroq
from langsmith import Client
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools

# Load environment variables
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


class ChatHistoryStore:
    """Manages chat history backed by Streamlit session state."""
    def __init__(self, session_key: str = "agent_messages"):
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
    
    def clear(self):
        """Clear all messages from history."""
        st.session_state[self.session_key] = []
# Setting Up Langchain Tracing
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

# Model dictionary mapping for API calls
model_dict = {
    "DeepSeek":"llama-3.1-8b-instant",
    "LLaMA 3.1-8B": "llama-3.1-8b-instant",
    "Gemma2 9B": "gemma2-9b-it",
    "Mixtral": "mixtral-8x7b-32768",
}

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

# Model initialization
selected_model = model_dict[st.session_state["model"]]

model = ChatGroq(
    model=selected_model,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=temperature,
    max_tokens=max_tokens,
    streaming=True,
)

# Main interface
st.title("Research Mate 🤖")
st.write("Your research-oriented assistant developed by Mitesh😎, ready to assist with academic and research queries!")

# Load tools and create agent
tools = load_tools(["arxiv"])
client = Client()
prompt = client.pull_prompt("hwchase17/react")
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True,early_stopping_method="force", max_iterations=3)

# Initialize chat history store
chat_history = ChatHistoryStore()

# Display previous chat messages
for msg in chat_history.get_messages():
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_input = st.chat_input("Ask a question...")

if user_input:
    chat_history.add_message("user", user_input)
    with st.chat_message("user"):
        st.write(user_input)

    response = agent_executor.invoke({"input": user_input})

    chat_history.add_message("assistant", response['output'])
    with st.chat_message("assistant"):
        st.write(response['output'])
    
else:
    with st.chat_message("assistant"): 
        st.write(greetings[st.session_state["language"]])
