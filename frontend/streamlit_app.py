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

if "active_document" not in st.session_state:
    st.session_state.active_document = None

if "messages" not in st.session_state:
    st.session_state.messages = []

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
                        st.session_state.active_document = data["filename"]
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

    st.subheader("Conversation")

    if st.button(
        "Clear conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


if st.session_state.active_document:

    st.markdown(
        f"""
        <div class="active-document">
            <strong>Active document</strong><br>
            {st.session_state.active_document}
        </div>
        """,
        unsafe_allow_html=True
    )

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
        "history": history
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

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
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