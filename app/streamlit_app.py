"""
Streamlit Chat Interface — Enterprise Knowledge Assistant.
Run with:  streamlit run streamlit_app.py
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ["GEMINI_API_KEY"] = ""
os.environ["GEMINI_MODEL"]   = "gemini-2.0-flash"

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🏢",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading knowledge base…")
def load_pipeline():
    from app.rag_pipeline import RAGPipeline
    pipeline   = RAGPipeline()
    sample_dir = os.getenv("SAMPLE_DOCS_DIR", "./data/sample_docs")
    if pipeline.collection_stats()["total_chunks"] == 0 and os.path.exists(sample_dir):
        pipeline.ingest_directory(sample_dir)
    return pipeline


@st.cache_resource(show_spinner="Loading AI model…")
def load_generator():
    from app.answer_generator import AnswerGenerator
    return AnswerGenerator()


try:
    pipeline  = load_pipeline()
    generator = load_generator()
except ValueError as exc:
    st.error(f"⚙️ Configuration error: {exc}")
    st.info(
        "Create a `.env` file in the same folder as `streamlit_app.py` with:\n\n"
        "```\nGEMINI_API_KEY=\n```\n\n"
        "Get a key at: https://aistudio.google.com/app/apikey"
    )
    st.stop()
except Exception as exc:
    st.error(f"❌ Startup error: {exc}")
    st.stop()


with st.sidebar:
    st.title("🏢 Knowledge Assistant")
    st.markdown("Ask questions about your internal documents.")
    st.divider()

    st.subheader("📄 Upload Document")
    st.caption("Supported: PDF, DOCX, TXT, MD  •  Max 200 MB")
    uploaded_file = st.file_uploader(
        "Drag & drop or browse",
        type=["pdf", "docx", "txt", "md"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        if st.button("Ingest Document", use_container_width=True):
            from app.rag_pipeline import EXTRACTORS
            suffix = Path(uploaded_file.name).suffix.lower()
            if suffix not in EXTRACTORS:
                st.error(f"Unsupported file type: {suffix}")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                with st.spinner(f"Indexing {uploaded_file.name}…"):
                    try:
                        count = pipeline.ingest_file(tmp_path)
                        st.success(f"✅ Added **{count}** chunks from `{uploaded_file.name}`")
                    except Exception as exc:
                        st.error(f"Ingestion failed: {exc}")
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)

    st.divider()
    stats = pipeline.collection_stats()
    st.metric("Indexed Chunks", stats["total_chunks"])
    st.caption(f"Model: {stats['embed_model']}")
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['document']}** — Page {s['page']}")
        if msg.get("confidence"):
            st.caption(f"Confidence: {msg['confidence']:.0%}")
        if msg.get("error"):
            st.warning(f"⚠️ {msg['error']}")


if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer…"):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if m["role"] in ("user", "assistant")
            ][-6:]

            answer     = ""
            sources    = []
            confidence = 0.0
            error      = None

            try:
                chunks = pipeline.retrieve(prompt, top_k=5)
                result = generator.generate(
                    question=prompt,
                    chunks=chunks,
                    conversation_history=history,
                )
                answer     = result["answer"]
                sources    = result.get("sources", [])
                confidence = result.get("confidence", 0.0)
                error      = result.get("error")
            except Exception as exc:
                answer = "⚠️ An unexpected error occurred."
                error  = str(exc)

        st.markdown(answer)
        if sources:
            with st.expander("📚 Sources"):
                for s in sources:
                    st.markdown(f"- **{s['document']}** — Page {s['page']}")
        if confidence:
            st.caption(f"Confidence: {confidence:.0%}")
        if error:
            st.warning(f"⚠️ {error}")

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "sources": sources, "confidence": confidence, "error": error,
    })