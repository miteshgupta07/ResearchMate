import streamlit as st

chat_page=st.Page(page="views/chat.py",
                 title="Chat",
                 icon=":material/chat:",
                 default=True)

about_app=st.Page(page="views/about_chatbot.py",
                  title="About Chatbot",
                  icon=":material/info:")

about_me=st.Page(page="views/about_developer.py",
                 title="About Developer",
                 icon=":material/person:")

pg=st.navigation(pages=[chat_page,about_app,about_me])

pg.run()