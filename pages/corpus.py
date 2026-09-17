import pandas as pd
import streamlit as st

from rag_core import (
    get_available_publication_years,
    get_corpus_documents,
    get_corpus_stats,
    get_document_chunks,
)

st.title("🔬 Scientific Literature Platform")
st.header("📚 Corpus Explorer")

# ---------------------------------------------------------------------
# Corpus summary
# ---------------------------------------------------------------------
try:
    summary = get_corpus_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", f"{summary['documents']:,}")
    col2.metric("Chunks", f"{summary['chunks']:,}")
    col3.metric("Embedded", f"{summary['embedded']:,}")
    col4.metric("Pending", f"{summary['pending']:,}")

except Exception as exc:
    st.error(f"Error while loading corpus statistics: {exc}")
    st.stop()

# ---------------------------------------------------------------------
# Search + year filters
# ---------------------------------------------------------------------
try:
    available_years = get_available_publication_years()
except Exception as exc:
    available_years = []
    st.warning(f"Could not load publication years from Supabase: {exc}")

year_options = ["All"] + available_years

search_col, from_col, to_col = st.columns([2, 1, 1])

with search_col:
    search_text = st.text_input(
        "Search documents",
        placeholder="Search by title, source URL or document ID...",
    ).strip()

with from_col:
    selected_from_year = st.selectbox(
        "From year",
        year_options,
        index=0,
        key="corpus_from_year_filter",
    )

with to_col:
    selected_to_year = st.selectbox(
        "To year",
        year_options,
        index=0,
        key="corpus_to_year_filter",
    )

year_from = (
    None
    if selected_from_year == "All"
    else int(selected_from_year)
)

year_to = (
    None
    if selected_to_year == "All"
    else int(selected_to_year)
)

if (
    year_from is not None
    and year_to is not None
    and year_from > year_to
):
    st.warning("From year cannot be greater than To year.")
    st.stop()

# ---------------------------------------------------------------------
# Documents: shown immediately when the page loads
# ---------------------------------------------------------------------
try:
    document_rows = get_corpus_documents(
        search_text=search_text,
        publication_year_from=year_from,
        publication_year_to=year_to,
    )
except Exception as exc:
    st.error(f"Error while loading corpus documents: {exc}")
    st.stop()

df = pd.DataFrame(
    document_rows,
    columns=[
        "document_id",
        "title",
        "source_url",
        "publication_year",
        "embedding_status",
        "chunk_count",
    ],
)

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "document_id": st.column_config.NumberColumn(
            "Document ID",
            format="%d",
        ),
        "title": st.column_config.TextColumn(
            "Title",
            width="large",
        ),
        "source_url": st.column_config.LinkColumn(
            "Source",
            display_text="Open article",
        ),
        "publication_year": st.column_config.NumberColumn(
            "Year",
            format="%d",
        ),
        "embedding_status": st.column_config.TextColumn(
            "Embedding Status",
        ),
        "chunk_count": st.column_config.NumberColumn(
            "Chunks",
            format="%d",
        ),
    },
)

if df.empty:
    st.info("No documents match the current search and year filters.")
    st.stop()

# ---------------------------------------------------------------------
# Document inspection
# ---------------------------------------------------------------------
st.subheader("Inspect document chunks")

labels = {
    int(row.document_id): (
        f"{int(row.document_id)} — {row.title}"
    )
    for row in df.itertuples()
}

selector_col, spacer = st.columns([2, 1])

with selector_col:
    document_id = st.selectbox(
        "Select a document",
        options=list(labels.keys()),
        format_func=lambda value: labels[value],
    )

selected_document = df.loc[
    df["document_id"] == document_id
].iloc[0]

meta_parts = []

if pd.notna(selected_document["publication_year"]):
    meta_parts.append(
        f"Year: {int(selected_document['publication_year'])}"
    )

meta_parts.append(
    f"Chunks: {int(selected_document['chunk_count'])}"
)

if pd.notna(selected_document["embedding_status"]):
    meta_parts.append(
        f"Embedding: {selected_document['embedding_status']}"
    )

st.caption(" · ".join(meta_parts))

if selected_document["source_url"]:
    st.link_button(
        "Open source article",
        selected_document["source_url"],
    )

try:
    chunk_rows = get_document_chunks(document_id)
except Exception as exc:
    st.error(f"Error while loading document chunks: {exc}")
    st.stop()

for position, chunk in enumerate(chunk_rows, start=1):
    section_title = (
        chunk.get("section_title")
        or "Untitled section"
    )

    chunk_index = chunk.get("chunk_index")
    display_index = (
        chunk_index
        if chunk_index is not None
        else position - 1
    )

    with st.expander(
        f"Chunk {display_index} | {section_title}"
    ):
        details = [
            f"chunk_id={chunk['chunk_id']}",
        ]

        if chunk.get("section_chunk_index") is not None:
            details.append(
                f"section_chunk={chunk['section_chunk_index']}"
            )

        content = chunk.get("content") or ""
        details.append(
            f"characters={len(content):,}"
        )

        st.caption(" · ".join(details))
        st.write(content)
