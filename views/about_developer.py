import streamlit as st

# Setting the title for the page
st.title("About Me 😎")

# Introduction Section
st.header("👋 Introduction")
st.write("""
Hello there! I'm Mitesh Gupta, a passionate Machine Learning enthusiast and developer.
I love building innovative AI solutions and exploring the latest advancements in deep learning, generative AI, and NLP.
Feel free to connect and explore my projects!
""")
col1,col2,col3 =st.columns([0.18,0.05,0.4], border=False,vertical_alignment="center")
with col1:
    st.link_button("Visit my portfolio here","https://miteshgupta-portfolio.vercel.app/")

with col2:
    st.write("**Or**")
with col3:
    st.download_button(
        label="Download My Resume",
        data="https://drive.google.com/file/d/1LKDQ5XjscIuGzvulEFBiBigT2f0U5k0R/view?usp=sharing",
        file_name="Mitesh_Gupta_Resume.pdf",
        mime="application/pdf",
    )
# Skills Section
st.header("🚀 Skills")
st.markdown("""
- **Programming Languages**: Python
- **AI/ML Frameworks**: TensorFlow, PyTorch, LangChain
- **Tools**: Streamlit, Hugging Face, OpenAI APIs
- **Databases**: MySQL, FAISS, ChromaDB
- **Other Skills**: Data Science, Machine Learning, Deep Learning, NLP, Generative AI, RAG Systems
""")

# Interests Section
st.header("🌟 Interests")
st.markdown("""
- **AI Research**: Staying updated with state-of-the-art AI models and research papers.
- **Projects**: Working on innovative tools like chatbots, dashboards, and RAG systems.
- **Community**: Sharing knowledge through blogs and contributing to open-source projects.
""")

# Contact Section
st.header("📬 Contact Me")
st.write("Thank you for visiting my chatbot application. Let’s build something amazing together!")

# Add links with icons for contact
st.markdown(
    '''
    <div style="display: flex; justify-content: center; gap: 20px;">
        <a href="mailto:miteshgupta2711@gmail.com" target="_blank">
            <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/65dea6c4eaca7da319e552c09f4cf5a9a8dab2c8/icons/Gmail-Dark.svg" alt="Gmail" width="100" height="100">
        </a>
        <a href="https://www.linkedin.com/in/mitesh-gupta/" target="_blank">
            <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/65dea6c4eaca7da319e552c09f4cf5a9a8dab2c8/icons/LinkedIn.svg" alt="LinkedIn" width="100" height="100">
        </a>
        <a href="https://x.com/mg_mitesh" target="_blank">
            <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/65dea6c4eaca7da319e552c09f4cf5a9a8dab2c8/icons/Twitter.svg" alt="Twitter" width="100" height="100">
        </a>
        <a href="https://www.instagram.com/mg_mitesh_gupta/" target="_blank">
            <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/65dea6c4eaca7da319e552c09f4cf5a9a8dab2c8/icons/Instagram.svg" alt="Instagram" width="100" height="100">
        </a>
        <a href="https://github.com/miteshgupta07" target="_blank">
            <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/65dea6c4eaca7da319e552c09f4cf5a9a8dab2c8/icons/Github-Light.svg" alt="GitHub" width="100" height="100">
        </a>
    </div>
    ''',
    unsafe_allow_html=True
)

