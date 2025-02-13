import streamlit as st

# App title
st.title("🔍 Research Mate")
st.subheader("Your AI-powered Research Assistant")

# Introduction
st.write("Search for research papers and get AI-generated summaries and insights.")

# Search Box
query = st.text_input("Enter a research topic or question:", "")

# Search Button
if st.button("Search"):
    st.write(f"🔎 Searching for: **{query}**")
    # Placeholder for search functionality
