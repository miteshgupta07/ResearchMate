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
Hi, I’m Mitesh Gupta an AI Engineer with strong expertise in Machine Learning, Deep Learning, NLP, and Generative AI.
Experienced in building and deploying scalable AI solutions and backend systems using Python and FastAPI.
Skilled in end-to-end model development, data processing, API design, and cloud deployment. Adept at translating complex business problems into practical, production-ready AI applications.
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
    with open("resume/Mitesh Gupta Resume-September 2026.pdf", "rb") as file:
        st.download_button(
            label="Download My Resume",
            data=file,
            file_name="Mitesh Gupta Resume.pdf",
            mime="application/pdf",
            icon="📄"
        )


# Skills Section
st.header("🚀 Skills")
st.markdown("""

- **Data Science & AI:** Python, Machine Learning, Deep Learning, CV, NLP, Generative AI, Feature Engineering, Model Optimization 

- **MLOps & Deployment:** AWS, Docker, Kubernetes , MLflow, FastAPI , Model Versioning, CI/CD Pipelines, Hugging Face, LLM Evaluation 

- **LLM & Agent Frameworks:** MCP, LangGraph, LangChain, TensorFlow, Keras, PyTorch , Scikit-learn, OpenCV, NLTK, spaCy, Streamlit 

- **Data Analysis & Visualization:** Data Preprocessing, Data Visualization, Data Analysis, SQL, Pandas, NumPy 

- **Databases & Vector Stores:** MySQL, PostgreSQL, Chroma, FAISS 

- **Version Control & Tools:** Git, GitHub 
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

