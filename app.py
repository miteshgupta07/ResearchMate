import streamlit as st

rag_page=st.Page(page="views/rag.py",
                 title="Chat",
                 icon=":material/chat:",
                 default=True)

about_me=st.Page(page="views/about_developer.py",
                 title="About Developer",
                 icon=":material/person:")


pg=st.navigation(pages=[rag_page,about_me])

pg.run()
