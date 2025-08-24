import streamlit as st
import os

# Page configuration
st.set_page_config(
    page_title='RAG-Powered Thesis Explorer',
    layout='wide',
    page_icon="🎓",
    initial_sidebar_state="collapsed"
)

# Configuration of the padding
st.html(
    """
    <style>
    .block-container {
        padding-top: 55px;
        padding-left: 0px;
        padding-right: 0px;
        padding-bottom: 0px;
        width: 60%;
    }
    </style>
    """
)

pg = st.navigation([
    st.Page(os.path.join('views', 'home.py'), title="Home", icon="🏠"),
    st.Page(os.path.join('views', 'rag.py'), title="Chat with AI (RAG)", icon="💬"),
    st.Page(os.path.join('views', 'paper.py'), title="Thesis Report", icon="📄"),
])
pg.run()
