import streamlit as st

# Setting the title for the page
st.title("About ResearchMate 🤖")

# Introduction Section
st.header("🌟 Introduction")
st.write("""
ResearchMate is a free AI research assistant developed by Mitesh Gupta, designed to assist users with academic and research queries. 
It leverages advanced AI models and agent-based architectures to provide accurate and well-researched responses.
With features like real-time retrieval from arXiv and the ability to query uploaded research papers, ResearchMate is a powerful tool for researchers and students alike.
""")

# Purpose Section
st.header("🎯 Purpose")
st.write("""
The main purposes of ResearchMate are to:
- Provide a conversational interface for interacting with AI-driven research tools.
- Enable users to upload research papers and ask queries directly.
- Utilize AI agents for real-time retrieval of academic papers from arXiv.
- Enhance research productivity by delivering precise and contextual insights.
""")

# Technologies Used Section
st.header("⚙️ Technologies Used")
st.write("""
ResearchMate is built using the following technologies:
- **Streamlit**: To create an interactive and user-friendly interface.
- **LangChain**: For managing chat history, prompt templates, and agent-based architectures.
- **FAISS**: For efficient vector-based search and retrieval.
- **Python**: As the primary programming language for backend logic.
- **arXiv API**: For real-time access to academic papers.
""")

# Features Section
st.header("🔑 Key Features")
st.markdown("""
- **Research Paper Querying**: Upload a paper and ask specific questions.
- **AI Agent for Real-Time Retrieval**: Fetch relevant research papers from arXiv.
- **Persistent Chat History**: Maintain the context of academic queries.
- **Multilingual Support**: Communicate in various languages, enhancing accessibility. 🌐
""")

# Benefits Section
st.header("💡 Benefits")
st.write("""
ResearchMate provides:
- Enhanced productivity for researchers and students.
- A streamlined interface for accessing and querying research materials.
- Real-time insights and academic support through AI-driven tools.
""")

# Closing Section
st.markdown("#### **ResearchMate is continually evolving to provide enhanced features and better support for its users. Explore its capabilities and accelerate your research journey!** 🌱")
