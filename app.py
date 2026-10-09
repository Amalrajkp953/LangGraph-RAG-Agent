
import streamlit as st

from rag.loader import load_pdf
from rag.embeddings import get_embeddings
from rag.vector_store import create_vector_store
from rag.llm import get_llm
from rag.graph import create_graph


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="StockLens | PDF Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap'
    );

    :root {
        --bg: #f6f8f4;
        --surface: #ffffff;
        --sidebar: #edf2e9;
        --green: #244c3a;
        --green-dark: #193b2f;
        --text: #26372d;
        --muted: #647368;
        --border: #dfe6dc;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar) !important;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    /* General text contrast */
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] li,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] small,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] {
        color: var(--text);
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: var(--text) !important;
    }

    /* Brand */
    .brand {
        font-family: 'Manrope', sans-serif;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -1.2px;
        color: var(--green-dark) !important;
        margin-bottom: 3px;
    }

    .brand-caption {
        font-size: 10px;
        letter-spacing: 1.5px;
        color: var(--muted) !important;
        margin-bottom: 28px;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #52735e !important;
        font-size: 11px;
        font-weight: 700;
    }

    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(32px, 4vw, 48px);
        font-weight: 800;
        color: var(--green-dark) !important;
        letter-spacing: -1.8px;
        line-height: 1.15;
        margin: 12px 0;
    }

    .hero-description {
        color: var(--muted) !important;
        font-size: 15px;
        line-height: 1.8;
        max-width: 700px;
    }

    .section-label {
        color: #53695a !important;
        font-weight: 700;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 25px 0 12px;
    }

    /* Document card */
    .document-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px;
        margin: 12px 0 16px;
    }

    .document-name {
        color: var(--green-dark) !important;
        font-weight: 700;
        font-size: 13px;
        overflow-wrap: anywhere;
    }

    .document-caption {
        color: var(--muted) !important;
        font-size: 11px;
        margin-top: 6px;
    }

    .status-text {
        color: var(--muted) !important;
        font-size: 12px;
        margin: 10px 0 16px;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 11px;
        min-height: 43px;
        border: 1px solid var(--border);
        font-weight: 600;
        color: var(--text) !important;
        background: #ffffff;
        transition: all 0.15s ease;
    }

    div.stButton > button:hover {
        border-color: #8aa391;
        color: var(--green-dark) !important;
    }

    div.stButton > button[kind="primary"] {
        background: var(--green) !important;
        border-color: var(--green) !important;
        color: #ffffff !important;
    }

    div.stButton > button[kind="primary"] p {
        color: #ffffff !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid var(--border);
        padding: 15px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: var(--green-dark) !important;
        -webkit-text-fill-color: var(--green-dark) !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 15px 18px;
        margin-bottom: 14px;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 {
        color: #26372d !important;
        -webkit-text-fill-color: #26372d !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid #d7e0d5 !important;
        border-radius: 15px !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #26372d !important;
        -webkit-text-fill-color: #26372d !important;
        background: #ffffff !important;
        caret-color: #244c3a !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #69786d !important;
        -webkit-text-fill-color: #69786d !important;
        opacity: 1 !important;
    }

    /* Expanders and source references */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] li {
        color: #26372d !important;
    }

    /* Alerts */
    [data-testid="stAlert"] p {
        color: #26372d !important;
    }

    /* Footer */
    .footer-note {
        text-align: center;
        color: #718074 !important;
        font-size: 11px;
        padding: 24px 0 10px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# SESSION STATE
# ==================================================

defaults = {
    "graph": None,
    "messages": [],
    "history": [],
    "document_pages": 0,
    "pending_question": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# BACKEND INITIALIZATION
# ==================================================

def initialize_rag():
    """Connect the frontend to the existing RAG backend."""

    with st.spinner("Loading your PDF and preparing the assistant..."):
        pages = load_pdf()
        embeddings = get_embeddings()
        vector_store = create_vector_store(pages, embeddings)
        llm = get_llm()
        graph = create_graph(vector_store, llm)

    st.session_state.graph = graph
    st.session_state.document_pages = len(pages)
    st.session_state.messages = []
    st.session_state.history = []


# ==================================================
# SOURCE FORMATTER
# ==================================================

def format_source(source):
    if isinstance(source, str):
        return source

    if isinstance(source, dict):
        filename = (
            source.get("filename")
            or source.get("source")
            or source.get("file")
        )

        page = source.get("page")
        if page is None:
            page = source.get("page_number")

        details = []

        if filename:
            details.append(str(filename))

        if page is not None:
            try:
                # LangChain PDF page metadata is usually zero-indexed.
                page_number = int(page)
                if source.get("page_is_one_indexed"):
                    display_page = page_number
                else:
                    display_page = page_number + 1

                details.append(f"Page {display_page}")
            except (ValueError, TypeError):
                details.append(f"Page {page}")

        return " · ".join(details) if details else str(source)

    return str(source)


# ==================================================
# ASK THE EXISTING LANGGRAPH
# ==================================================

def ask_question(question):
    result = st.session_state.graph.invoke(
        {
            "question": question,
            "context": "",
            "answer": "",
            "history": st.session_state.history,
            "sources": [],
        }
    )

    answer = result.get(
        "answer",
        "I couldn't generate an answer for this question."
    )

    sources = result.get("sources", [])

    if isinstance(sources, (str, dict)):
        sources = [sources]

    st.session_state.history = result.get(
        "history",
        st.session_state.history,
    )

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": str(answer),
        "sources": sources or [],
    })


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            StockLens<span style="color:#82a18b;">.</span>
        </div>
        <div class="brand-caption">
            DOCUMENT INTELLIGENCE WORKSPACE
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Your workspace")

    st.markdown(
        """
        <div class="document-card">
            <div style="font-size:25px;margin-bottom:8px;">📄</div>
            <div class="document-name">
                Stock_Market_Performance_2024.pdf
            </div>
            <div class="document-caption">
                Configured source document · PDF
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.graph is not None:
        st.success("Assistant ready")

        st.caption(
            f"PDF pages loaded: {st.session_state.document_pages}"
        )
    else:
        st.markdown(
            '<div class="status-text">⚪ Assistant not initialized</div>',
            unsafe_allow_html=True,
        )

    if st.button(
        "Initialize assistant",
        type="primary",
        use_container_width=True,
    ):
        try:
            initialize_rag()
            st.rerun()
        except Exception as error:
            st.error("Could not initialize the assistant.")
            st.exception(error)

    st.divider()

    st.markdown("#### Quick actions")

    if st.button(
        "＋ New conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown("#### About StockLens")

    st.caption(
        "Ask questions about your configured PDF. "
        "The existing LangGraph pipeline retrieves relevant "
        "content and generates document-grounded answers."
    )

    st.divider()

    st.caption("STOCKLENS · LANGGRAPH RAG")


# ==================================================
# MAIN HEADER
# ==================================================

st.markdown(
    '<div class="eyebrow">YOUR DOCUMENT, UNDERSTOOD</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-title">Ask your document<br>anything.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-description">
        Explore the 2024 stock market report using natural language.
        Ask about market trends, explore a company's performance,
        and ask follow-up questions using your document.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ==================================================
# METRICS
# ==================================================

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Workspace", "StockLens")

with metric2:
    page_count = (
        str(st.session_state.document_pages)
        if st.session_state.document_pages
        else "Not loaded"
    )
    st.metric("PDF pages", page_count)

with metric3:
    question_count = sum(
        1 for message in st.session_state.messages
        if message["role"] == "user"
    )
    st.metric("Questions asked", question_count)


# ==================================================
# SUGGESTED QUESTIONS
# ==================================================

if not st.session_state.messages:
    st.markdown(
        '<div class="section-label">Start with a suggested question</div>',
        unsafe_allow_html=True,
    )

    suggestions = [
        ("📊 Market overview", "What is this PDF about?"),
        ("🍎 Apple performance", "What was Apple's performance in 2024?"),
        ("🟢 Nvidia analysis", "What does the PDF say about Nvidia?"),
        ("📈 Key takeaways", "What were the main stock market trends in 2024?"),
    ]

    col1, col2 = st.columns(2)

    for index, (label, question_text) in enumerate(suggestions):
        column = col1 if index % 2 == 0 else col2

        with column:
            if st.button(
                label,
                key=f"suggestion_{index}",
                use_container_width=True,
            ):
                if st.session_state.graph is None:
                    st.warning(
                        "Initialize the assistant in the sidebar first."
                    )
                else:
                    st.session_state.pending_question = question_text
                    st.rerun()


# ==================================================
# CHAT HISTORY
# ==================================================

if st.session_state.messages:
    st.markdown(
        '<div class="section-label">Conversation</div>',
        unsafe_allow_html=True,
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            sources = message.get("sources", [])

            if message["role"] == "assistant" and sources:
                with st.expander("📚 View sources"):
                    for source in sources:
                        st.markdown(f"- {format_source(source)}")


# ==================================================
# CHAT INPUT
# ==================================================

if st.session_state.graph is None:
    st.info(
        "Select **Initialize assistant** in the sidebar to start. "
        "Your existing PDF, vector store, embeddings, LLM, and "
        "LangGraph workflow will be used."
    )

else:
    typed_question = st.chat_input(
        "Ask a question about your stock market PDF..."
    )

    question = typed_question

    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question and question.strip():
        try:
            with st.spinner(
                "Searching the document and preparing your answer..."
            ):
                ask_question(question.strip())

            st.rerun()

        except Exception as error:
            st.error("Something went wrong while answering your question.")
            st.exception(error)


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div class="footer-note">
        StockLens · Document-grounded answers · LangGraph RAG
    </div>
    """,
    unsafe_allow_html=True,
)