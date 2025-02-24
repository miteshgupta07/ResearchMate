import streamlit as st

rag_page=st.Page(page="views/rag.py",
                 title="Chat",
                 icon=":material/chat:",
                 default=True)

agent_page=st.Page(page="views/agent.py",
                   title="Agent",
                   icon=":material/smart_toy:")

about_app=st.Page(page="views/about_chatbot.py",
                  title="About Chatbot",
                  icon=":material/info:")

about_me=st.Page(page="views/about_developer.py",
                 title="About Developer",
                 icon=":material/person:")

pg=st.navigation(pages=[rag_page,agent_page,about_app,about_me])

pg.run()
