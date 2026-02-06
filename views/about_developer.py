import streamlit as st

# Scoped CSS for layout customization
st.markdown(
    """
    <style>
    .block-container {
        padding-left: 10rem;
        padding-right: 10rem;
        max-width: 1400px;
    }
    .content-wrapper {
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Setting the title for the page
st.markdown(
    "<h1 style='text-align: center;'>About Me 😎</h1>",
    unsafe_allow_html=True
)

# Introduction Section
st.header("👋 Introduction")
st.write("""
Hi, I’m Mitesh Gupta—an AI Engineer focused on building real-world, scalable intelligence.
I work across machine learning, deep learning, generative AI, and NLP to turn complex ideas into production-ready systems.
Explore my work to see how I approach modern AI problems.
""")
import streamlit as st

# Inject CSS
st.markdown("""
<style>
/* Link button */
div[data-testid="stLinkButton"] a[data-testid^="stBaseLinkButton"] {
    background-color: #FF0000;
    color: white;
    border: none;
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
    text-decoration: none;
}

/* Hover */
div[data-testid="stLinkButton"] a[data-testid^="stBaseLinkButton"]:hover {
    background-color: #333333;
    color: white;
}

/* Download button */
div[data-testid="stDownloadButton"] > button {
    background-color: #FF0000;
    color: white;
    border: none;
}
div[data-testid="stDownloadButton"] > button:hover {
    background-color: #333333;
    color: white;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([0.27, 0.05, 1.0], vertical_alignment="center")

with col1:
    st.link_button(
        "Visit my portfolio here",
        "https://miteshgupta-portfolio.vercel.app/",
        icon="💻"
    )

with col2:
    st.write("**Or**")

with col3:
    st.link_button(
        "Download My Resume",
        "https://drive.google.com/file/d/1DNFjKVUIK0V2l98ituAnhNNQ70IIpIsj/view?usp=sharing",
        icon="📄"
    )


# Skills Section
st.header("🚀 Skills")
st.markdown("""

- **Programming & Data Foundations:** Python, Data Science, Pandas, NumPy

- **Machine Learning & Deep Learning:** Machine Learning, Deep Learning, Scikit-learn, TensorFlow, Keras, PyTorch

- **AI Specializations:** Generative AI, Natural Language Processing (NLP), Computer Vision, Retrieval-Augmented Generation (RAG) Systems

- **Frameworks & NLP Tooling:** LangChain, NLTK, spaCy, OpenCV

- **Cloud, MLOps & Deployment:** AWS, Docker, MLflow, CI/CD Pipelines, Streamlit, Groq, Hugging Face

- **Databases & Vector Stores:** MySQL, Vector Databases: ChromaDB, FAISS
""")

# Interests Section
st.header("🌟 Interests")
st.markdown("""
- **AI Research**: Staying updated with state-of-the-art AI models and research papers.
- **Projects**: Working on innovative tools like production grade chatbots and RAG systems.
- **Community**: Sharing knowledge through blogs and contributing to open-source projects.
""")

# Contact Section
st.header("📬 Contact Me")
st.write("Thank you for visiting my chatbot application. Let’s build something amazing together!")

# Add links with icons for contact
st.markdown(
    '''
    <div style="display: flex; gap: 20px;">
        <a href="mailto:miteshgupta2711@gmail.com" target="_blank">
            <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/65dea6c4eaca7da319e552c09f4cf5a9a8dab2c8/icons/Gmail-Dark.svg" alt="Gmail" width="80" height="80">
        </a>
        <a href="https://www.linkedin.com/in/mitesh-gupta/" target="_blank">
            <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/65dea6c4eaca7da319e552c09f4cf5a9a8dab2c8/icons/LinkedIn.svg" alt="LinkedIn" width="80" height="80">
        </a>
        <a href="https://x.com/mg_mitesh" target="_blank">
            <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/65dea6c4eaca7da319e552c09f4cf5a9a8dab2c8/icons/Twitter.svg" alt="Twitter" width="80" height="80">
        </a>
        <a href="https://www.instagram.com/mg_mitesh_gupta/" target="_blank">
            <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/65dea6c4eaca7da319e552c09f4cf5a9a8dab2c8/icons/Instagram.svg" alt="Instagram" width="80" height="80">
        </a>
        <a href="https://github.com/miteshgupta07" target="_blank">
            <img src="https://raw.githubusercontent.com/tandpfun/skill-icons/65dea6c4eaca7da319e552c09f4cf5a9a8dab2c8/icons/Github-Light.svg" alt="GitHub" width="80" height="80">
        </a>
    </div>
    ''',
    unsafe_allow_html=True
)

