import streamlit as st

pages = {
    "LLM Powered Apps": [
        st.Page("bot.py", title="Chatbot Powered by TinyLLM"),
        st.Page("email.py", title="Email Generator Powered by TinyLLM"),
    ],
}

pg = st.navigation(pages)
pg.run()
