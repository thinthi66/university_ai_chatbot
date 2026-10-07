import streamlit as st
import requests
import os

page_icon = "logo.png" if os.path.exists("uom_logo.png") else "U"
# Page settings
st.set_page_config(
    page_title="University of Mauritius AI Assistant",
    page_icon= page_icon,
    layout="centered"
)

# Sidebar
with st.sidebar:
    st.title("UoM AI Assistant")
    st.markdown("Ask questions about the University of Mauritius.")
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show assistant reply
    with st.chat_message("assistant"):
        st.markdown("**Backend not connected yet.**\n\nThis is a placeholder. Real answers will appear here later.")

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Backend not connected yet. This is a placeholder."
    })