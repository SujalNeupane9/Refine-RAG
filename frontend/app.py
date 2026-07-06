import os
import logging
from io import BytesIO

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
logger = logging.getLogger(__name__)


st.set_page_config(page_title="RefineRAG", layout="wide")
st.title("RefineRAG")

st.session_state.setdefault("answer_payload", None)
st.session_state.setdefault("ingest_payload", None)
st.session_state.setdefault("debug_payload", {})
st.session_state.setdefault("uploaded_filename", None)


def ingest_document(uploaded_file):
    if uploaded_file is None:
        return None

    logger.info("Uploading document for ingestion: %s", uploaded_file.name)
    files = {
        "file": (
            uploaded_file.name,
            BytesIO(uploaded_file.getvalue()),
            uploaded_file.type or "application/octet-stream",
        )
    }
    response = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=120)
    response.raise_for_status()
    payload = response.json()
    logger.info("Document ingestion completed: %s", payload.get("message"))
    return payload


def ask_question(question: str):
    response = requests.post(
        f"{BACKEND_URL}/ask",
        json={"question": question},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def show_request_error(action: str, exc: requests.RequestException) -> None:
    if isinstance(exc, requests.ConnectionError):
        st.error(f"{action} failed because the backend is unavailable.")
    elif isinstance(exc, requests.Timeout):
        st.error(f"{action} timed out while waiting for the backend.")
    elif getattr(exc, "response", None) is not None:
        response = exc.response
        detail = response.text.strip() or "No error details were returned."
        st.error(f"{action} failed with status {response.status_code}.")
        st.caption(detail)
    else:
        st.error(f"{action} failed.")


st.subheader("Upload Knowledge Documents")
uploaded_file = st.file_uploader(
    "Upload a PDF, TXT, or Markdown file",
    type=["pdf", "txt", "md"],
)

if uploaded_file is not None:
    st.caption(f"Selected file: {uploaded_file.name}")

upload_col, question_col = st.columns(2)

with upload_col:
    ingest_clicked = st.button("Ingest Document", use_container_width=True)
    if ingest_clicked:
        if uploaded_file is None:
            st.warning("Choose a file first.")
        else:
            with st.spinner("Ingesting document..."):
                try:
                    st.session_state.ingest_payload = ingest_document(uploaded_file)
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.success(st.session_state.ingest_payload["message"])
                    logger.info("Ingested file stored in session: %s", uploaded_file.name)
                except requests.RequestException as exc:
                    show_request_error("Document ingestion", exc)
                    st.session_state.debug_payload["ingest_error"] = str(exc)
                    logger.exception("Document ingestion failed")

with question_col:
    question = st.text_input("Ask a question", placeholder="What is RefineRAG?")
    ask_clicked = st.button("Ask Question", use_container_width=True)
    if ask_clicked:
        if not question.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    st.session_state.answer_payload = ask_question(question.strip())
                    st.session_state.debug_payload["last_question"] = question.strip()
                    st.success("Answer ready.")
                except requests.RequestException as exc:
                    show_request_error("Question answering", exc)
                    st.session_state.debug_payload["ask_error"] = str(exc)
                    logger.exception("Question answering failed")

st.subheader("Answer")
answer_payload = st.session_state.get("answer_payload")
if answer_payload:
    st.write(answer_payload.get("answer", ""))
else:
    st.info("Ask a question to see the generated answer here.")

st.subheader("Sources")
if answer_payload and answer_payload.get("sources"):
    for source in answer_payload["sources"]:
        source_bits = []
        if source.get("file"):
            source_bits.append(source["file"])
        if source.get("page") is not None:
            source_bits.append(f"page {source['page']}")
        if source.get("chunk_text") is not None:
            source_bits.append(f"chunk {source['chunk_text']}")
        st.write(" - " + ", ".join(source_bits) if source_bits else " - Unlabeled source")
else:
    st.info("Sources will appear here after an answer is generated.")

st.subheader("Critique")
if answer_payload and answer_payload.get("critique"):
    st.write(answer_payload.get("critique"))
else:
    st.info("The critique will appear here after an answer is generated.")

st.subheader("Debug")
with st.expander("Debug / Critique", expanded=False):
    st.write("Backend URL:", BACKEND_URL)
    if st.session_state.get("uploaded_filename"):
        st.write("Last uploaded file:", st.session_state["uploaded_filename"])
    if st.session_state.get("ingest_payload"):
        st.json(st.session_state["ingest_payload"])
    if st.session_state.get("debug_payload"):
        st.json(st.session_state["debug_payload"])
