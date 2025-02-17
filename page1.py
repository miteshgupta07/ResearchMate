import os
import streamlit as st
from langchain.tools import WikipediaQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from groq import Groq

# Initialize LangChain's arXiv tool
arxiv_tool = ArxivQueryRun()

def search_research_papers(query):
    """Retrieve research paper links from arXiv."""
    return arxiv_tool.run(query)

# Streamlit UI
st.title("📚 Search Research Papers & Download")
st.write("Enter a topic to find related research papers from arXiv.")

query = st.text_input("🔎 Enter a research topic:", "")
url = "https://www.streamlit.io"
st.write("check out this [link](%s)" % url)
st.markdown("check out this [link](%s)" % url)
if st.button("Search"):
    # Fetch research papers from arXiv
    st.subheader("📄 Research Paper Links")
    arxiv_results = search_research_papers(query)

    # Display results as clickable links
    if arxiv_results:
        paper_links = arxiv_results.split("\n")  # Assuming results are newline-separated
        for link in paper_links:
            st.write("🔗 [link](%s)" % url)
    else:
        st.write("❌ No papers found. Try a different topic.")
