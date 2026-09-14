import os
import streamlit as st
import requests

st.set_page_config(
    page_title="Document Intelligence Assistant",
    page_icon="📄",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        opacity: 0.7;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Local default; the Docker image sets this so the UI can find the API inside the container
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

if "active_document_id" not in st.session_state:
    st.session_state.active_document_id = None   # None = search all documents

if "messages" not in st.session_state:
    st.session_state.messages = []


def fetch_documents():
    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        if response.status_code == 200:
            return response.json()["documents"]
    except requests.exceptions.RequestException:
        pass
    return []


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)} passages)"):
        for i, source in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {source['filename']} — page {source['page']}**")
            st.caption(source["snippet"] + "…")

st.markdown(
    '<div class="app-title">Document Intelligence Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">'
    'Upload a PDF and ask grounded questions using hybrid retrieval and reranking.'
    '</div>',
    unsafe_allow_html=True
)
with st.sidebar:
    st.header("Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.button("Upload & Index"):

            with st.spinner("Processing document..."):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf"
                    )
                }

                try:
                    response = requests.post(
                        f"{API_BASE_URL}/documents",
                        files=files
                    )

                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.active_document_id = data["document_id"]
                        st.success(
                            f"{data['filename']} uploaded successfully"
                        )

                    else:
                        st.error(
                            f"Upload failed: {response.text}"
                        )

                except requests.exceptions.RequestException:
                    st.error(
                        "Could not connect to the FastAPI backend."
                    )


    st.divider()

    st.subheader("Documents")

    documents = fetch_documents()

    # Dropdown: "All documents" or one specific upload (the id keeps duplicates apart)
    labels = ["All documents"] + [
        f"{doc['filename']} ({doc['document_id'][:8]})" for doc in documents
    ]
    ids = [None] + [doc["document_id"] for doc in documents]

    current = ids.index(st.session_state.active_document_id) \
        if st.session_state.active_document_id in ids else 0

    choice = st.selectbox("Search in", labels, index=current)
    st.session_state.active_document_id = ids[labels.index(choice)]

    if st.session_state.active_document_id:
        if st.button("Delete this document", use_container_width=True):
            requests.delete(
                f"{API_BASE_URL}/documents/{st.session_state.active_document_id}"
            )
            st.session_state.active_document_id = None
            st.rerun()

    st.divider()

    st.subheader("Conversation")

    if st.button(
        "Clear conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


if st.session_state.active_document_id:
    st.caption(f"Searching in: {choice}")
elif documents:
    st.caption("Searching in: all uploaded documents")
else:
    st.caption("No document uploaded yet")

st.subheader("Chat with your document")

if not st.session_state.messages:
    st.info(
        "Upload a document and ask a question to begin."
    )

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources"))

question = st.chat_input("Ask a question about your document...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    history = [
        {
            "role": message["role"],
            "content": message["content"]
        }
        for message in st.session_state.messages[:-1]
    ]

    payload = {
        "question": question,
        "history": history,
        "document_id": st.session_state.active_document_id
    }

    try:

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = requests.post(
                    f"{API_BASE_URL}/chat",
                    json=payload
                )

                if response.status_code == 200:

                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])

                    st.markdown(answer)
                    show_sources(sources)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        }
                    )

                else:
                    st.error(
                        f"Chat failed: {response.text}"
                    )

    except requests.exceptions.RequestException:

        st.error(
            "Could not connect to the FastAPI backend."
        )