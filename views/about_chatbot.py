import streamlit as st

# Setting the title for the page
st.title("About the Chatbot 🤖")

# Introduction Section
st.header("🌟 Introduction")
st.write("""
This chatbot is a powerful tool designed to assist users with general queries. 
It uses state-of-the-art conversational AI models to provide accurate and context-aware responses.
The chatbot is equipped with a persistent history feature to maintain the flow of the conversation and ensure a seamless user experience. 
Additionally, the chatbot supports multiple languages, allowing users from diverse backgrounds to interact effortlessly.
""")

# Purpose Section
st.header("🎯 Purpose")
st.write("""
The main purpose of this chatbot is to:
- Offer a conversational interface for interacting with advanced language models.
- Enable users to ask questions in their preferred language.
- Provide a platform for exploring the capabilities of cutting-edge AI models in a user-friendly manner.
""")

# Technologies Used Section
st.header("⚙️ Technologies Used")
st.write("""
The chatbot is built using the following technologies:
- **Streamlit**: To create a simple yet interactive user interface.
- **ChatGroq**: For utilizing advanced conversational AI models.
- **LangChain**: For managing chat history and prompt templates.
- **Python**: As the backbone programming language for logic and integrations.
- **Groq Models**: To deliver precise, creative, and contextually relevant responses.
""")

# Features Section
st.header("🔑 Key Features")
st.markdown("""
- **Multi-Model Support**: Choose from different models (e.g., LLaMA, Gemma2, Mixtral).
- **Adjustable Parameters**: Fine-tune the chatbot's behavior by customizing temperature and max tokens.
- **Persistent Chat History**: Maintain the context of the conversation across queries.
- **Multilingual Support**: Communicate in languages like English, Hindi, Spanish, French, and German. 🌐
""")

# Benefits Section
st.header("💡 Benefits")
st.write("""
This chatbot provides:
- Easy interaction with sophisticated AI systems.
- Multilingual capabilities to make the chatbot accessible to a global audience.
- A user-friendly interface for both general queries and specific use cases.
""")

# Closing Section
st.markdown("#### **This chatbot is continually evolving to provide enhanced features and better support for its users. Explore its capabilities and enjoy the experience!** 🌱")
