import streamlit as st

from rag_core import get_groq_model

st.set_page_config(
    page_title="Scientific Literature Platform",
    page_icon="🔬",
    layout="wide",
)

pages = [
    st.Page(
        "pages/chat.py",
        title="Chat",
        icon=":material/chat:",
        default=True,
    ),
    st.Page(
        "pages/corpus.py",
        title="Corpus Explorer",
        icon=":material/library_books:",
    ),
    st.Page(
        "pages/evaluation.py",
        title="Retrieval Evaluation",
        icon=":material/analytics:",
    ),
]

navigation = st.navigation(
    pages,
    position="sidebar",
)

with st.sidebar:
    st.caption("System")
    st.write("Embedding model: `all-MiniLM-L6-v2`")
    st.write("Vector store: `Supabase + pgvector`")
    st.write("Retrieval: `Semantic + Lexical + RRF`")
    st.write("Reranker: `ms-marco-MiniLM-L-6-v2`")
    st.write("LLM provider: `Groq`")
    st.write(f"LLM model: `{get_groq_model()}`")

    st.markdown("---")
    st.caption("Developed by:")
    st.markdown("**Luis José Bolaños Berrocal**")

navigation.run()
