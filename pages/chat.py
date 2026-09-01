import streamlit as st

from rag_core import (
    generate_grounded_answer,
    get_available_publication_years,
    get_db_connection,
    get_groq_client,
    load_embedding_model,
    load_reranker,
    retrieve_by_strategy,
    retrieval_strategy_description,
    source_dataframe,
    translate_query_for_retrieval,
)

st.title("🔬 Scientific Literature Platform")
st.header("💬 Ask Literature")
st.caption("Ask questions against the indexed scientific literature.")


def evidence_column_config(active_strategy):
    config = {
        "Source": st.column_config.TextColumn("Source"),
        "Document ID": st.column_config.NumberColumn("Document ID", format="%d"),
        "Chunk ID": st.column_config.NumberColumn("Chunk ID", format="%d"),
        "Year": st.column_config.NumberColumn("Year", format="%d"),
    }

    if active_strategy == "Vector":
        config["Semantic score"] = st.column_config.NumberColumn(
            "Semantic score",
            format="%.4f",
            help="Cosine similarity from vector search. Higher values rank better; this is not a probability.",
        )
        config["Semantic rank"] = st.column_config.NumberColumn(
            "Semantic rank",
            format="%d",
            help="Position returned by the vector search.",
        )

    elif active_strategy == "Lexical":
        config["Lexical score"] = st.column_config.NumberColumn(
            "Lexical score",
            format="%.4f",
            help="PostgreSQL full-text search ranking score. Higher values rank better; this is not a probability.",
        )
        config["Lexical rank"] = st.column_config.NumberColumn(
            "Lexical rank",
            format="%d",
            help="Position returned by the lexical search.",
        )

    elif active_strategy == "Hybrid":
        config["RRF score"] = st.column_config.NumberColumn(
            "RRF score",
            format="%.6f",
            help="Reciprocal Rank Fusion score (k=60). It combines semantic and lexical ranks; it is not a probability.",
        )
        config["Semantic rank"] = st.column_config.NumberColumn("Semantic rank", format="%d")
        config["Lexical rank"] = st.column_config.NumberColumn("Lexical rank", format="%d")

    elif active_strategy == "Hybrid + Reranking":
        config["Reranker score"] = st.column_config.NumberColumn(
            "Reranker score",
            format="%.4f",
            help="CrossEncoder relevance score used to reorder RRF candidates; it is not a probability.",
        )
        config["RRF score"] = st.column_config.NumberColumn(
            "RRF score",
            format="%.6f",
            help="Reciprocal Rank Fusion score (k=60). It combines semantic and lexical ranks; it is not a probability.",
        )
        config["Semantic rank"] = st.column_config.NumberColumn("Semantic rank", format="%d")
        config["Lexical rank"] = st.column_config.NumberColumn("Lexical rank", format="%d")

    return config

strategy = st.selectbox(
    "Retrieval strategy",
    [
        "Hybrid + Reranking",
        "Hybrid",
        "Vector",
        "Lexical",
    ],
    index=0,
)

st.caption(f"Pipeline: {retrieval_strategy_description(strategy)}")

st.subheader("Search filters")

try:
    available_years = get_available_publication_years()
except Exception as exc:
    available_years = []
    st.warning(f"Could not load publication years from Supabase: {exc}")

year_options = ["All"] + available_years

year_col1, year_col2, author_col = st.columns([1, 1, 2])

with year_col1:
    selected_from_year = st.selectbox(
        "From year",
        year_options,
        index=0,
        key="from_year_filter",
    )

with year_col2:
    selected_to_year = st.selectbox(
        "To year",
        year_options,
        index=0,
        key="to_year_filter",
    )

with author_col:
    author_filter = st.text_input(
        "Author contains",
        placeholder="e.g. Smith",
    ).strip()

year_from = None if selected_from_year == "All" else int(selected_from_year)
year_to = None if selected_to_year == "All" else int(selected_to_year)

if year_from is not None and year_to is not None and year_from > year_to:
    st.warning("From year cannot be greater than To year.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("strategy"):
            st.caption(f"Retrieval: {message['strategy']}")

        if message.get("sources"):
            with st.expander("Retrieved evidence"):
                st.dataframe(
                    source_dataframe(
                        message["sources"],
                        message.get("strategy"),
                    ),
                    use_container_width=True,
                    hide_index=True,
                    column_config=evidence_column_config(
                        message.get("strategy")
                    ),
                )

question = st.chat_input(
    "Ask a question about the indexed scientific literature"
)

if question:
    if year_from is not None and year_to is not None and year_from > year_to:
        st.warning("Please correct the publication year range first.")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
            "strategy": strategy,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            embedding_model = load_embedding_model()
            groq_client = get_groq_client()

            # Only load the CrossEncoder when the selected strategy needs it.
            reranker = None
            if strategy == "Hybrid + Reranking":
                reranker = load_reranker()

            with st.status(
                f"Running {strategy} retrieval...",
                expanded=False,
            ) as status:
                retrieval_query = translate_query_for_retrieval(
                    groq_client,
                    question,
                )

                connection = get_db_connection()
                try:
                    retrieval = retrieve_by_strategy(
                        connection=connection,
                        embedding_model=embedding_model,
                        reranker=reranker,
                        retrieval_query=retrieval_query,
                        strategy=strategy,
                        publication_year_from=year_from,
                        publication_year_to=year_to,
                        author_contains=author_filter or None,
                    )
                finally:
                    connection.close()

                sources = retrieval["final"]

                response = generate_grounded_answer(
                    client=groq_client,
                    original_question=question,
                    retrieval_query=retrieval_query,
                    reranked_results=sources,
                )

                status.update(
                    label=f"{strategy} retrieval completed",
                    state="complete",
                    expanded=False,
                )

            st.caption(f"Retrieval strategy: {strategy}")
            st.markdown(response)

            if sources:
                with st.expander("Retrieved evidence"):
                    st.dataframe(
                        source_dataframe(sources, strategy),
                        use_container_width=True,
                        hide_index=True,
                        column_config=evidence_column_config(strategy),
                    )

                    for index, item in enumerate(sources, start=1):
                        st.markdown(
                            f"**[S{index}] {item.get('section_title') or 'Untitled section'} "
                            f"— Document {item['document_id']}**"
                        )
                        metadata = [
                            f"Year: {item.get('publication_year') or 'N/A'}",
                            f"Journal: {item.get('journal') or 'N/A'}",
                        ]
                        if item.get("reranker_score") is not None:
                            metadata.append(
                                f"Reranker: {item.get('reranker_score'):.4f}"
                            )
                        st.caption(" · ".join(metadata))
                        st.write(item.get("content") or "")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "sources": sources,
                    "strategy": strategy,
                }
            )

        except Exception as exc:
            st.error(f"Error while running the RAG pipeline: {exc}")
