import os
import sys
import requests
import streamlit as st

st.set_page_config(page_title="RAG Pipeline", page_icon="📚", layout="wide")

API_URL = "http://localhost:8000"

st.title("📚 RAG Pipeline")

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

# Tabs
tab_chat, tab_compare = st.tabs(["Chat", "Retrieval Comparison"])

# ============ CHAT TAB ============
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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

# ============ COMPARISON TAB ============
with tab_compare:
    st.subheader("Compare Retrieval Strategies")
    compare_query = st.text_input("Enter a query to compare:", placeholder="e.g. What is the habitable zone?")

    if compare_query and st.button("Compare"):
        with st.spinner("Running all strategies..."):
            try:
                resp = requests.post(f"{API_URL}/query/debug", json={"question": compare_query})
                if resp.status_code != 200:
                    st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
                else:
                    data = resp.json()

                    # Answer
                    st.markdown("### Answer")
                    st.markdown(data["answer"])

                    st.divider()

                    # Side-by-side comparison
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Vector Search")
                        for i, r in enumerate(data["debug"]["vector"], 1):
                            with st.expander(f"#{i} | Score: {r['score']:.4f} | Page {r['page']}"):
                                st.text(r["text"])

                    with col2:
                        st.markdown("#### BM25 (Keyword)")
                        for i, r in enumerate(data["debug"]["bm25"], 1):
                            with st.expander(f"#{i} | Score: {r['score']:.3f} | Page {r['page']}"):
                                st.text(r["text"])

                    st.divider()

                    col3, col4 = st.columns(2)

                    with col3:
                        st.markdown("#### Hybrid (RRF Merged)")
                        for i, r in enumerate(data["debug"]["hybrid"], 1):
                            with st.expander(f"#{i} | RRF: {r['score']:.4f} | Page {r['page']} | via: {r['retrieval']}"):
                                st.text(r["text"])

                    with col4:
                        st.markdown("#### Parent-Child")
                        for i, r in enumerate(data["debug"]["parent_child"], 1):
                            with st.expander(f"#{i} | Score: {r['score']:.4f} | Page {r['page']}"):
                                st.markdown("**Child (search match):**")
                                st.text(r["child"])
                                if r.get("parent"):
                                    st.markdown("**Parent (full context):**")
                                    st.text(r["parent"])

            except requests.ConnectionError:
                st.error("API not running.")
