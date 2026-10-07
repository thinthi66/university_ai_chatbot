import streamlit as st
import requests
import os

# Page icon
page_icon = "logo.png" if os.path.exists("logo.png") else "U"

# Page settings
st.set_page_config(
    page_title="University of Mauritius AI Assistant",
    page_icon=page_icon,
    layout="centered"
)

# Backend URL
BACKEND_URL = "http://127.0.0.1:8000/chat"

# Sidebar
with st.sidebar:
    st.title("UoM AI Assistant")
    st.markdown("Ask questions about the University of Mauritius.")
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

# Store chat history & session id
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for s in message["sources"]:
                    title = s.get("title", "Source")
                    url = s.get("url", "")
                    if url:
                        st.markdown(f"- [{title}]({url})")
                    else:
                        st.markdown(f"- {title}")

# Chat input
if prompt := st.chat_input("Ask a question..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")

        try:
            response = requests.post(
                BACKEND_URL,
                json={
                    "question": prompt,
                    "session_id": st.session_state.session_id
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer received.")
                sources = data.get("sources", [])
                st.session_state.session_id = data.get("session_id")

                placeholder.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            title = s.get("title", "Source")
                            url = s.get("url", "")
                            if url:
                                st.markdown(f"- [{title}]({url})")
                            else:
                                st.markdown(f"- {title}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            else:
                error_msg = f"Server error ({response.status_code}). Please try again."
                placeholder.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

        except requests.exceptions.ConnectionError:
            error_msg = (
                "**Backend is not running.**\n\n"
                "Start it with:\n"
                "`cd backend` → `uvicorn app.main:app --reload`"
            )
            placeholder.markdown(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            placeholder.markdown(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })