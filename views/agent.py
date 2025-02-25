# Importing necessary modules
import streamlit as st
import os
from dotenv import load_dotenv

# LangChain & AI Model Imports
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain.agents.agent import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
# Load environment variables
load_dotenv()

# Setting Up Langchain Tracing
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

# Model dictionary mapping for API calls
model_dict = {
    "DeepSeek":"deepseek-r1-distill-qwen-32b",
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
prompt = hub.pull("hwchase17/react")
neutral_prompt = ChatPromptTemplate.from_template(
    "Maintain the conversation context based on the provided agent_messages and respond appropriately."
)
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True,early_stopping_method="force", max_iterations=3)

# Initialize chat history if not present
if "agent_chat_history" not in st.session_state:
    st.session_state["agent_chat_history"] = ChatMessageHistory()

if "agent_messages" not in st.session_state:
    st.session_state["agent_messages"] = []

def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    return st.session_state["agent_chat_history"]

# Display previous chat agent_messages
for msg in st.session_state["agent_messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state["agent_messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    chain = neutral_prompt | model

    with_message_history = RunnableWithMessageHistory(
        chain,
        get_chat_history,
        input_messages_key="agent_messages",
    )

    intermediate_response = with_message_history.invoke(
        {
            "language": st.session_state.language,
            "agent_messages": [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.agent_messages],
        },
        config={"configurable": {"session_id": "default_agent_session"}},
    )

    response = agent_executor.invoke({"input": user_input})

    st.session_state.agent_messages.append({"role": "assistant", "content": response['output']})
    with st.chat_message("assistant"):
        st.write(response['output'])
    
else:
    with st.chat_message("assistant"): 
        st.write(greetings[st.session_state["language"]])
