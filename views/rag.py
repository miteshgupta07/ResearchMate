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
model = ChatGroq(
    model=selected_model,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=temperature,
    max_tokens=max_tokens,
    streaming=True,
)

# Setting up the main Streamlit interface and initializing the chatbot UI
st.title("IntelliChat 🤖")
st.write("Your intelligent assistant developed by Mitesh😎, ready to answer your queries!")


# Initializing session state variables for chat history and user-assistant messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()  # Manages message history within the session

if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to retrieve session-specific chat history for maintaining conversation context
def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    return st.session_state.chat_history  # Returns the chat history for the current session

# Creating a prompt template to structure the assistant's responses and define its behavior
generic_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant developed by Mitesh. Answer all questions to the best of your ability and give response in given {language} language."),
        MessagesPlaceholder(variable_name="messages"),  # Dynamically includes conversation history
    ]
)

# Combining the language model and chat history for managing conversation flow
chain = generic_template | model
with_message_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,  # Function to fetch chat history
    input_messages_key="messages",  # Key to access the messages
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

    # Generating the assistant's response based on the chat history and input
    response = with_message_history.invoke(
        {"language": st.session_state.language,
         "messages": [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]},
        config={"configurable": {"session_id": "default_session"}},
    )

    # Storing and displaying the assistant's response in the chat history
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    with st.chat_message("assistant"):
        st.write(response.content)

else:
    # Adding a welcome message at the start of the session
    with st.chat_message(""): 
        st.write(greetings[st.session_state.language])