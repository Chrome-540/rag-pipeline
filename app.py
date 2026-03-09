import os
import sys
import requests
import streamlit as st

st.set_page_config(page_title="RAG Pipeline", page_icon="📚", layout="wide")

API_URL = "http://localhost:8000"

st.title("📚 RAG Pipeline")
st.caption("Upload documents and ask questions")

# Sidebar — Document Management
with st.sidebar:
    st.header("Documents")

    # Upload
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file and st.button("Ingest"):
        with st.spinner("Ingesting... (this may take a minute)"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            resp = requests.post(f"{API_URL}/ingest", files=files)

        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Ingested! {data['total_chunks']} chunks created.")
        else:
            st.error(f"Failed: {resp.json().get('detail', 'Unknown error')}")

    # List documents
    st.divider()
    try:
        docs = requests.get(f"{API_URL}/documents").json()["documents"]
        if docs:
            for doc in docs:
                col1, col2 = st.columns([3, 1])
                col1.write(f"📄 {doc['filename']} ({doc['size_kb']} KB)")
                if col2.button("🗑️", key=doc["filename"]):
                    requests.delete(f"{API_URL}/documents/{doc['filename']}")
                    st.rerun()
        else:
            st.info("No documents uploaded yet.")
    except requests.ConnectionError:
        st.error("API not running. Start it with:\nuvicorn src.main:app --reload")

# Main — Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(f"{API_URL}/query", json={"question": question})
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]

                    # Show sources
                    if data.get("sources"):
                        sources_text = "\n".join(
                            f"- {s['source']}, Page {s['page']}" + (f" ({s['section']})" if s.get('section') else "")
                            for s in data["sources"]
                        )
                        answer += f"\n\n---\n**Sources:**\n{sources_text}"

                    st.markdown(answer)
                else:
                    answer = f"Error: {resp.json().get('detail', 'Unknown error')}"
                    st.error(answer)
            except requests.ConnectionError:
                answer = "API not running. Start the FastAPI server first."
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
