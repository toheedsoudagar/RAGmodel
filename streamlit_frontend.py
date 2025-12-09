# streamlit_frontend.py — lightweight Streamlit UI that calls the backend /rag endpoint
import streamlit as st
import requests
import os

st.set_page_config(page_title="RAG Chat", layout="centered")
st.title("RAG Chat (Frontend)")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")
# Use environment variable or Streamlit secrets in hosted mode
API_KEY = os.environ.get("BACKEND_API_KEY", "") or st.secrets.get("BACKEND_API_KEY", "")

if not API_KEY:
    st.warning("No BACKEND_API_KEY set — set BACKEND_API_KEY in environment or Streamlit Secrets.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant", "content":"Connected to remote RAG backend."}]

# Render chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Input
user_input = st.chat_input("Ask a question...")
if user_input:
    st.session_state.messages.append({"role":"user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    with st.spinner("Contacting backend..."):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/rag",
                json={"q": user_input},
                headers={"X-Api-Key": API_KEY, "Content-Type": "application/json"},
                timeout=120
            )
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "No answer returned.")
                st.session_state.messages.append({"role":"assistant", "content": answer})
                st.chat_message("assistant").markdown(answer)
            else:
                st.session_state.messages.append({"role":"assistant", "content": f"Error {resp.status_code}: {resp.text}"})
                st.chat_message("assistant").markdown(f"Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.session_state.messages.append({"role":"assistant", "content": f"Request failed: {e}"})
            st.chat_message("assistant").markdown(f"Request failed: {e}")
