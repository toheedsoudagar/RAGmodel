# newapp.py
import streamlit as st
from rag import RAGPipeline 

st.set_page_config(page_title="Local RAG Chat", layout="centered")
st.title("Offline RAG")

# --- Pipeline Initialization ---
@st.cache_resource
def initialize_pipeline():
    return RAGPipeline()

pipeline = initialize_pipeline()

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Initial greeting message
if not st.session_state.messages:
    # Check if SQL is available to adjust the greeting
    db_setup_status = "and is ready for quantitative questions about your SQL data." if pipeline.sql_agent.agent_executor else "but the SQL agent is currently disabled."
    st.session_state.messages.append({"role": "assistant", "content": f"Hello! I'm your hybrid RAG assistant. The document ingestion is complete {db_setup_status}"})


# --- Message Rendering ---
def render_message(role, content):
    avatar = "👤" if role == "user" else "✨"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content) 

for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])

# --- Chat Input and Response ---
query = st.chat_input("Ask a question...")

if query:
    # 1. Display user query
    st.session_state.messages.append({"role": "user", "content": query})
    render_message("user", query)

    # 2. Get RAG/SQL answer
    with st.spinner("Thinking..."):
        answer, docs = pipeline.ask(query)

    final_answer = answer 

    # 3. Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    render_message("assistant", final_answer)