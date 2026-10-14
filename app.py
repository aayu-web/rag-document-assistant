import os
import time
import uuid
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()



APP_NAME = "Groundwork"
MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TEXTBOOK_PATH = os.path.join("document-loader", "dl.pdf")
TEXTBOOK_DB_PATH = "./chroma_db"
TEXTBOOK_COLLECTION = "deep_learning_docs"

CANDIDATE_K = 12
FINAL_K = 4

PROMPT = ChatPromptTemplate.from_template("""
You are a careful document research assistant.

Answer the user's question using only the supplied context.
Use clear, natural language and organize the answer when helpful.

Rules:
- Do not invent facts that are not supported by the context.
- If the documents do not contain enough information, say so.
- Cite relevant sources using [Source 1], [Source 2], etc.
- Distinguish what the documents explicitly say from any reasonable
  interpretation of their contents.
- Do not claim to have read information that is not in the context.

Context:
{context}

Question:
{question}

Answer:
""")




st.set_page_config(
    page_title="Groundwork | Document Research",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    :root {
        --page: #f7f6f2;
        --surface: #ffffff;
        --ink: #262b27;
        --muted: #71766f;
        --line: #e4e2da;
        --accent: #385747;
        --accent-hover: #2c4638;
    }

    .stApp {
        background: var(--page);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(247, 246, 242, 0.92);
    }

    [data-testid="stSidebar"] {
        background: #f0efe9;
        border-right: 1px solid var(--line);
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--ink) !important;
        letter-spacing: -0.035em;
    }

    p, li {
        line-height: 1.7;
    }

    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    .brand {
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.045em;
        color: var(--accent);
        margin-bottom: 0;
    }

    .eyebrow {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        font-weight: 650;
    }

    .intro {
        color: #60665f;
        font-size: 1.02rem;
        max-width: 690px;
        line-height: 1.8;
    }

    .quiet-panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 1rem 1.1rem;
    }

    div[data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        padding-top: 0.7rem;
        padding-bottom: 0.7rem;
    }

    div[data-testid="stChatInput"] {
        border-color: #d8d8cf;
    }

    div[data-testid="stChatInput"] textarea {
        background: #fff;
    }

    .stButton > button {
        border-radius: 8px;
        min-height: 2.55rem;
        font-weight: 550;
        transition: background-color 0.15s ease;
    }

    .stButton > button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
        color: white;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--accent-hover);
        border-color: var(--accent-hover);
    }

    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 10px;
    }

    hr {
        border-color: var(--line);
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)



DEFAULTS = {
    "groq_llm": None,
    "uploaded_store": None,
    "uploaded_names": [],
    "chat_history": [],
    "textbook_store": None,
    "indexed_file_signature": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value




@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


@st.cache_resource(show_spinner="Loading the document reranker...")
def get_reranker():
    # Import lazily to avoid loading PyTorch at initial page render.
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)


@st.cache_resource
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
    )




def build_uploaded_index(uploaded_files):
    """Read and index arbitrary text-based PDFs in memory."""

    if not uploaded_files:
        raise ValueError("Upload at least one PDF first.")

    documents = []
    total_files = len(uploaded_files)

    progress = st.progress(0, text="Reading your documents...")

    for index, uploaded_file in enumerate(uploaded_files):
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".pdf", delete=False
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            pages = PyPDFLoader(temp_path).load()

            
            for page in pages:
                page.metadata["source"] = uploaded_file.name

            pages_with_text = [
                page for page in pages
                if page.page_content.strip()
            ]

            if not pages_with_text:
                st.warning(
                    f"No selectable text was found in "
                    f"{uploaded_file.name}. It may be a scanned PDF "
                    "that needs OCR."
                )

            documents.extend(pages_with_text)

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        progress.progress(
            (index + 1) / total_files,
            text=f"Reading document {index + 1} of {total_files}...",
        )

    if not documents:
        progress.empty()
        raise ValueError(
            "No readable text was found. Try a text-based PDF."
        )

    progress.progress(0.0, text="Splitting text into searchable passages...")

    chunks = get_text_splitter().split_documents(documents)

    if not chunks:
        progress.empty()
        raise ValueError("No searchable text chunks could be created.")

    progress.progress(0.0, text=f"Indexing {len(chunks):,} passages...")

   
    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=f"uploads_{uuid.uuid4().hex}",
    )

    progress.progress(1.0, text="Your documents are ready.")
    time.sleep(0.3)
    progress.empty()

    return store, len(documents), len(chunks)




def get_textbook_store():
    if st.session_state.textbook_store is not None:
        return st.session_state.textbook_store

    if os.path.isdir(TEXTBOOK_DB_PATH):
        try:
            store = Chroma(
                collection_name=TEXTBOOK_COLLECTION,
                embedding_function=get_embeddings(),
                persist_directory=TEXTBOOK_DB_PATH,
            )

            if store._collection.count() > 0:
                st.session_state.textbook_store = store
                return store

        except Exception as exc:
            st.warning(f"Could not open the textbook index: {exc}")

    return None




def get_page_number(doc):
    page_label = doc.metadata.get("page_label")

    if page_label:
        return str(page_label)

    page = doc.metadata.get("page")

    if page is not None:
        return str(int(page) + 1)

    return "Unknown"


def retrieve_and_rerank(question, source_mode):
    stores = []

    if source_mode in ("Your documents", "All available documents"):
        if st.session_state.uploaded_store is not None:
            stores.append(st.session_state.uploaded_store)

    if source_mode in ("Bundled textbook", "All available documents"):
        textbook_store = get_textbook_store()
        if textbook_store is not None:
            stores.append(textbook_store)

    if not stores:
        raise ValueError(
            "No indexed documents are available. Upload and index PDFs first, "
            "or select the bundled textbook if its index exists."
        )

    candidates = []

    for store in stores:
        candidates.extend(
            store.similarity_search(question, k=CANDIDATE_K)
        )

    # Remove duplicate passages while preserving document identity.
    unique_docs = []
    seen = set()

    for doc in candidates:
        signature = (
            doc.metadata.get("source", ""),
            get_page_number(doc),
            doc.page_content,
        )

        if signature not in seen:
            seen.add(signature)
            unique_docs.append(doc)

    if not unique_docs:
        return []

    reranker = get_reranker()

    pairs = [
        (question, doc.page_content)
        for doc in unique_docs
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(unique_docs, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return ranked[:FINAL_K]




def answer_question(question, source_mode):
    if st.session_state.groq_llm is None:
        raise ValueError(
            "Connect your Groq API key in the sidebar before asking a question."
        )

    ranked_docs = retrieve_and_rerank(question, source_mode)

    if not ranked_docs:
        return (
            "I couldn't find relevant passages in the selected documents.",
            [],
        )

    context_parts = []
    sources = []

    for rank, (doc, score) in enumerate(ranked_docs, start=1):
        filename = os.path.basename(
            doc.metadata.get("source", "Document")
        )
        page = get_page_number(doc)

        context_parts.append(
            f"[Source {rank}: {filename}, page {page}]\n"
            f"{doc.page_content}"
        )

        sources.append({
            "rank": rank,
            "filename": filename,
            "page": page,
            "score": float(score),
            "excerpt": doc.page_content,
        })

    response = st.session_state.groq_llm.invoke(
        PROMPT.format_messages(
            question=question,
            context="\n\n".join(context_parts),
        )
    )

    return response.content, sources




with st.sidebar:
    st.markdown(
        f'<p class="brand">{APP_NAME}</p>',
        unsafe_allow_html=True,
    )
    st.caption("A private-by-design document workspace")

    st.divider()

    st.markdown("#### Language model")
    st.caption(
        "Connect your own Groq API key to generate answers. "
        "Your key is used for this app session."
    )

    api_key = st.text_input(
        "Groq API key",
        type="password",
        key="groq_key_input",
        placeholder="gsk_...",
    )

    if st.button(
        "Connect",
        type="primary",
        use_container_width=True,
    ):
        if not api_key.strip():
            st.error("Enter your Groq API key first.")
        else:
            try:
                with st.spinner("Connecting to Groq..."):
                    candidate_llm = ChatGroq(
                        model=MODEL_NAME,
                        api_key=api_key.strip(),
                        temperature=0,
                        max_tokens=2048,
                    )
                    candidate_llm.invoke(
                        "Reply with the single word OK."
                    )

                st.session_state.groq_llm = candidate_llm
                st.success("Connected successfully.")

            except Exception:
                st.session_state.groq_llm = None
                st.error(
                    "Connection failed. Check your API key, "
                    "Groq account access, and model availability."
                )

    if st.session_state.groq_llm is not None:
        st.caption("Connection active for this session.")

    if st.button("Disconnect and clear session", use_container_width=True):
        st.session_state.groq_llm = None
        st.session_state.uploaded_store = None
        st.session_state.uploaded_names = []
        st.session_state.chat_history = []
        st.session_state.textbook_store = None
        st.session_state.indexed_file_signature = None
        st.rerun()

    st.divider()

    st.markdown("#### Documents")
    source_mode = st.radio(
        "Search within",
        options=[
            "Your documents",
            "Bundled textbook",
            "All available documents",
        ],
        index=0,
    )

    if st.session_state.uploaded_names:
        st.caption(
            f"{len(st.session_state.uploaded_names)} uploaded PDF(s) indexed"
        )

    st.divider()

    st.caption(
        "Uploaded indexes are kept in application session memory "
        "and are not intentionally persisted. This is not a guarantee "
        "of immediate memory erasure or a security boundary."
    )


# --------------------------------------------------
# Main workspace
# --------------------------------------------------

st.markdown(
    '<p class="eyebrow">DOCUMENT RESEARCH WORKSPACE</p>',
    unsafe_allow_html=True,
)

st.title("Understand your documents.")
st.markdown(
    '<p class="intro">Bring together reports, research papers, notes, '
    'manuals or books. Ask a question, follow the evidence, and return '
    'to the source when you need more detail.</p>',
    unsafe_allow_html=True,
)

tab_chat, tab_documents = st.tabs(
    ["Ask questions", "Manage documents"]
)




with tab_documents:
    st.subheader("Your document collection")
    st.write(
        "Upload one or more text-based PDFs. You don't need to specify "
        "their subject or category."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload research papers, reports, manuals, notes, or books.",
        key="pdf_upload",
    )

    if uploaded_files:
        st.caption(
            f"{len(uploaded_files)} PDF(s) selected · "
            f"{sum(f.size for f in uploaded_files) / (1024 * 1024):.1f} MB total"
        )

        for file in uploaded_files:
            st.write(f"📄 {file.name}")

    if st.button(
        "Process selected PDFs",
        type="primary",
        disabled=not uploaded_files,
    ):
        try:
            with st.spinner(
                "Reading, splitting, and indexing your documents..."
            ):
                new_store, page_count, chunk_count = build_uploaded_index(
                    uploaded_files
                )

            st.session_state.uploaded_store = new_store
            st.session_state.uploaded_names = [
                f.name for f in uploaded_files
            ]
            st.session_state.indexed_file_signature = [
                (f.name, f.size) for f in uploaded_files
            ]
            st.session_state.chat_history = []

            st.success(
                f"Ready to search {len(uploaded_files)} PDF(s), "
                f"{page_count:,} pages with readable text, "
                f"and {chunk_count:,} text passages."
            )

        except Exception as exc:
            st.error(f"Document processing failed: {exc}")

    if st.session_state.uploaded_names:
        st.markdown("**Currently indexed PDFs**")
        for name in st.session_state.uploaded_names:
            st.write(f"📄 {name}")

        if st.button("Remove uploaded documents"):
            st.session_state.uploaded_store = None
            st.session_state.uploaded_names = []
            st.session_state.indexed_file_signature = None
            st.session_state.chat_history = []
            st.rerun()

    st.divider()

    st.subheader("Bundled textbook")
    if os.path.isfile(TEXTBOOK_PATH):
        st.write("The bundled deep-learning textbook is available locally.")
        if get_textbook_store() is not None:
            st.success("Its existing search index is available.")
        else:
            st.info(
                "Its existing Chroma index was not found. "
                "The bundled textbook is not automatically re-indexed here."
            )
    else:
        st.caption("No bundled textbook is configured. You can still use uploaded PDFs.")



with tab_chat:
    if st.session_state.groq_llm is None:
        st.info(
            "Connect your Groq API key in the sidebar to start asking questions."
        )

    if (
        source_mode == "Your documents"
        and st.session_state.uploaded_store is None
    ):
        st.info(
            "Open **Manage documents** to upload and process your PDFs."
        )

    if (
        source_mode == "Bundled textbook"
        and get_textbook_store() is None
    ):
        st.info(
            "The bundled textbook index isn't available. "
            "You can upload any PDF under Manage documents instead."
        )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("Sources and supporting passages"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**Source {source['rank']} — "
                            f"{source['filename']}, page {source['page']}**"
                        )
                        st.caption(
                            f"Reranker score: {source['score']:.3f}"
                        )
                        st.write(source["excerpt"])
                        st.divider()

    question = st.chat_input(
        "Ask a question about your documents...",
        disabled=st.session_state.groq_llm is None,
    )

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your documents..."):
                    answer, sources = answer_question(
                        question,
                        source_mode,
                    )

                st.markdown(answer)

                if sources:
                    with st.expander("Sources and supporting passages"):
                        for source in sources:
                            st.markdown(
                                f"**Source {source['rank']} — "
                                f"{source['filename']}, "
                                f"page {source['page']}**"
                            )
                            st.caption(
                                f"Reranker score: {source['score']:.3f}"
                            )
                            st.write(source["excerpt"])
                            st.divider()

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as exc:
                error_message = (
                    f"I couldn't complete that request: {exc}"
                )
                st.error(error_message)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                })