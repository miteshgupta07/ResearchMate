# Research Mate: Your Free AI Research Assistant 📚🤖

**Research Mate** is a powerful AI research assistant designed to streamline your academic and research work. It offers advanced features such as real-time retrieval of research papers, intelligent summarization, and insightful suggestions for similar research papers, making it an indispensable tool for researchers and students alike.

---

## 🚀 Features
- **Upload and Query Research Papers:** Easily upload research papers (PDFs) and ask specific questions to get concise answers.
- **Real-Time Retrieval from arXiv:** Utilize AI agents to search and fetch relevant research papers directly from arXiv.
- **Research Paper Summarization:** Generate quick and accurate summaries of research papers.
- **Similar Research Paper Suggestions:** Get recommendations for papers related to your query or uploaded document.
- **Persistent Chat History:** Maintain the context of your research queries for a seamless experience.

---

## 🌍 Deployment
Research Mate can be accessed on [**Streamlit**](https://researchmate-chatbot.streamlit.app/) for a smooth and interactive user experience.

---

## 🛠️ How It Works
1. **Upload a Research Paper:** Users can upload PDF files of research papers.
2. **Query-Based Interaction:** Ask specific questions related to the uploaded document.
3. **AI Agent Utilization:** The AI agents handle:
   - **Summarization:** Providing concise summaries of research papers.
   - **Paper Search:** Retrieving research papers from arXiv based on the query.
   - **Suggestions:** Recommending similar research papers for deeper insights.
4. **Persistent History:** The chat history feature ensures research continuity.

---

## 📝 Usage Guide
1. **Clone the repository:**
```bash
git clone https://github.com/miteshgupta07/ResearchMate.git
cd ResearchMate
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```
3. **Set up API keys:**
Add your API keys to repository secrets (for deployment) or .env (for local use).

4. **Run the app locally:**
```bash
streamlit run app.py
```

## Technologies Used 📊 
- **Streamlit**: For an interactive user interface.
- **LangChain**: To manage prompts and chat history.
- **FAISS**: For efficient retrieval-augmented generation (RAG) processes.
- **ChatGroq**: High-performance conversational models.
- **Python**: Core programming language for logic and integration.

## Security 🛡️
API keys are securely stored as GitHub Repository Secrets or in .env files for local development.

## Contributing 🤝 
Contributions are welcome! Please open an issue or submit a pull request for any improvements or feature suggestions.


## License 📝
This project is licensed under the MIT License - see the [LICENSE](https://github.com/miteshgupta07/ResearchMate/blob/main/LICENSE) file for details.



## Acknowledgments 🙏 
- **LangChain**: For robust history and prompt management.
- **Groq Models**: For providing state-of-the-art conversational AI capabilities.
- **Streamlit Community**: For helpful resources and support.
- **FAISS**: For enabling fast and accurate research retrieval.
- **ArXiv**: For access to a vast repository of research papers.

## Contact
For inquiries or collaborations, please contact me at [miteshgupta2711@gmail.com](mailto:miteshgupta2711@gmail.com).
