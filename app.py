import streamlit as st
from global_rag import ask_global_rag, ingest_pdf
import os

st.set_page_config(page_title="Global Medical AI Assistant", page_icon="🌐")
st.title("🌐 Worldwide Medical RAG System")
st.caption("Powered by Pinecone Vector DB, HuggingFace, & Groq (Llama-3)")

# Sidebar PDF Uploader
with st.sidebar:
    st.header("📚 Document Ingestion")
    uploaded_file = st.file_uploader("Upload Medical PDF", type=["pdf"])
    if uploaded_file and st.button("Process & Sync to Cloud DB"):
        with st.spinner("Embedding and syncing to Pinecone..."):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            ingest_pdf(temp_path)
            os.remove(temp_path)
            st.success("Synced to Global Vector DB successfully!")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching global database & generating response..."):
            answer, citations = ask_global_rag(prompt)

            response_text = answer
            if citations:
                response_text += "\n\n📌 **Sources:** " + ", ".join(citations)

            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})