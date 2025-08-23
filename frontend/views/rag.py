import streamlit as st
from services.llm_service import get_llm_response

st.title("💬 Chat with AI (RAG)")

# Initialise the message history in the session status if it does not exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages from the history every time the app is reloaded
with st.container(height=490):
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

with st.container():
    # Capture user input
    if prompt := st.chat_input("¿En qué puedo ayudarte?"):
        # Add the user's message to the history and display it in the UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Responding... 🤔"):
            # Assistant's response
            assistant_response = get_llm_response(prompt)
            
            # Add the assistant's response to the history and display it in the UI
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        st.rerun()