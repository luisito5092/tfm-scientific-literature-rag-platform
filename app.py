import os
from typing import Optional

import pandas as pd
import pg8000
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# 14 - Streamlit RAG App
# ============================================================
# Generic UI: intentionally NOT hard-coded to a single scientific domain.
# When the corpus is replaced, the same app can be reused.
#
# Pipeline:
# User question
#   -> Groq translation to English
#   -> Semantic + Lexical retrieval
#   -> Reciprocal Rank Fusion (RRF)
#   -> Cross-Encoder reranking
#   -> Top 5 context
#   -> Groq grounded answer
# ============================================================


st.set_page_config(
    page_title="Scientific Literature RAG",
    page_icon="🔬",
    layout="wide",
)


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SEMANTIC_TOP_K = 20
LEXICAL_TOP_K = 20
HYBRID_TOP_K = 10
RERANKED_TOP_K = 5
RRF_K = 60


# ============================================================
# Configuration helpers
# ============================================================

def get_setting(name: str, default=None):
    """
    Read first from Streamlit secrets, then from environment variables.
    This makes the same app usable locally and on Streamlit Community Cloud.
    """
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name, default)


def require_setting(name: str):
    value = get_setting(name)

    if value is None or str(value).strip() == "":
        raise RuntimeError(
            f"Missing required configuration: {name}. "
            "Define it in .streamlit/secrets.toml or as an environment variable."
        )

    return value


def get_groq_model():
    return get_setting("GROQ_MODEL", "openai/gpt-oss-20b")


# ============================================================
# Cached models
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner="Loading reranker...")
def load_reranker():
    return CrossEncoder(RERANKER_MODEL_NAME)


# ============================================================
# Connections
# ============================================================

def get_db_connection():
    return pg8000.connect(
        host=require_setting("SUPABASE_HOST"),
        port=int(get_setting("SUPABASE_PORT", "6543")),
        database=get_setting("SUPABASE_DATABASE", "postgres"),
        user=require_setting("SUPABASE_USER"),
        password=require_setting("SUPABASE_PASSWORD"),
        ssl_context=True,
    )


def get_groq_client():
    return Groq(api_key=require_setting("GROQ_API_KEY"))


# ============================================================
# Query preprocessing
# ============================================================

def translate_query_for_retrieval(client, user_question):
    response = client.chat.completions.create(
        model=get_groq_model(),
        temperature=0,
        max_tokens=200,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a query preprocessing component for a scientific "
                    "literature retrieval system. Translate the user's question "
                    "into English while preserving its original scientific meaning. "
                    "If it is already English, return it unchanged. "
                    "Do not answer the question. Do not add facts, synonyms, "
                    "keywords, or explanations. Preserve names of diseases, genes, "
                    "compounds, treatments, nutrients, biomarkers, and technical terms. "
                    "Return only the English query."
                ),
            },
            {
                "role": "user",
                "content": user_question,
            },
        ],
    )

    translated = response.choices[0].message.content.strip()
    return translated or user_question


# ============================================================
# Retrieval
# ============================================================

def embedding_to_pgvector(embedding):
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def semantic_search(
    connection,
    embedding_model,
    question,
    top_k=SEMANTIC_TOP_K,
    publication_year_from: Optional[int] = None,
    publication_year_to: Optional[int] = None,
    author_contains: Optional[str] = None,
):
    embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
    )
    vector = embedding_to_pgvector(embedding)

    sql = """
    SELECT
        c.chunk_id,
        c.document_id,
        c.section_title,
        d.publication_year,
        d.authors,
        d.journal,
        c.content,
        1 - (c.embedding <=> CAST(%s AS vector)) AS semantic_score
    FROM rag.document_chunks c
    JOIN rag.documents d
        ON d.document_id = c.document_id
    WHERE
        c.embedding IS NOT NULL
        AND c.embedding_status = 'completed'
    """

    params = [vector]

    if publication_year_from is not None:
        sql += "\nAND d.publication_year >= %s"
        params.append(int(publication_year_from))

    if publication_year_to is not None:
        sql += "\nAND d.publication_year <= %s"
        params.append(int(publication_year_to))

    if author_contains:
        sql += "\nAND CAST(d.authors AS text) ILIKE %s"
        params.append(f"%{author_contains}%")

    sql += """
    ORDER BY c.embedding <=> CAST(%s AS vector)
    LIMIT %s
    """

    params.extend([vector, int(top_k)])

    cursor = connection.cursor()

    try:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
    finally:
        cursor.close()

    columns = [
        "chunk_id",
        "document_id",
        "section_title",
        "publication_year",
        "authors",
        "journal",
        "content",
        "semantic_score",
    ]

    return [dict(zip(columns, row)) for row in rows]


def lexical_search(
    connection,
    question,
    top_k=LEXICAL_TOP_K,
    publication_year_from: Optional[int] = None,
    publication_year_to: Optional[int] = None,
    author_contains: Optional[str] = None,
):
    search_vector = """
    (
        setweight(
            to_tsvector('english', coalesce(c.section_title, '')),
            'A'
        )
        ||
        setweight(
            to_tsvector('english', coalesce(c.content, '')),
            'B'
        )
    )
    """

    sql = f"""
    WITH q AS (
        SELECT websearch_to_tsquery('english', %s) AS query
    )
    SELECT
        c.chunk_id,
        c.document_id,
        c.section_title,
        d.publication_year,
        d.authors,
        d.journal,
        c.content,
        ts_rank_cd(
            {search_vector},
            q.query
        ) AS lexical_score
    FROM rag.document_chunks c
    JOIN rag.documents d
        ON d.document_id = c.document_id
    CROSS JOIN q
    WHERE {search_vector} @@ q.query
    """

    params = [question]

    if publication_year_from is not None:
        sql += "\nAND d.publication_year >= %s"
        params.append(int(publication_year_from))

    if publication_year_to is not None:
        sql += "\nAND d.publication_year <= %s"
        params.append(int(publication_year_to))

    if author_contains:
        sql += "\nAND CAST(d.authors AS text) ILIKE %s"
        params.append(f"%{author_contains}%")

    sql += """
    ORDER BY lexical_score DESC, c.chunk_id
    LIMIT %s
    """

    params.append(int(top_k))

    cursor = connection.cursor()

    try:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
    finally:
        cursor.close()

    columns = [
        "chunk_id",
        "document_id",
        "section_title",
        "publication_year",
        "authors",
        "journal",
        "content",
        "lexical_score",
    ]

    return [dict(zip(columns, row)) for row in rows]


def reciprocal_rank_fusion(
    semantic_results,
    lexical_results,
    rrf_k=RRF_K,
    final_top_k=HYBRID_TOP_K,
):
    fused = {}

    def ensure(item):
        chunk_id = item["chunk_id"]

        if chunk_id not in fused:
            fused[chunk_id] = {
                "chunk_id": item["chunk_id"],
                "document_id": item["document_id"],
                "section_title": item["section_title"],
                "publication_year": item["publication_year"],
                "authors": item["authors"],
                "journal": item["journal"],
                "content": item["content"],
                "semantic_rank": None,
                "semantic_score": None,
                "lexical_rank": None,
                "lexical_score": None,
                "rrf_score": 0.0,
            }

        return fused[chunk_id]

    for rank, item in enumerate(semantic_results, start=1):
        record = ensure(item)
        record["semantic_rank"] = rank
        record["semantic_score"] = float(item["semantic_score"])
        record["rrf_score"] += 1.0 / (rrf_k + rank)

    for rank, item in enumerate(lexical_results, start=1):
        record = ensure(item)
        record["lexical_rank"] = rank
        record["lexical_score"] = float(item["lexical_score"])
        record["rrf_score"] += 1.0 / (rrf_k + rank)

    ranked = sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    return ranked[:final_top_k]


def build_reranker_text(item):
    section_title = item.get("section_title") or ""
    content = item.get("content") or ""

    if section_title:
        return f"Section: {section_title}\n\n{content}"

    return content


def rerank_results(
    reranker,
    question,
    hybrid_results,
    final_top_k=RERANKED_TOP_K,
):
    if not hybrid_results:
        return []

    pairs = [
        [question, build_reranker_text(item)]
        for item in hybrid_results
    ]

    scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
    )

    reranked = []

    for hybrid_rank, (item, score) in enumerate(
        zip(hybrid_results, scores),
        start=1,
    ):
        result = dict(item)
        result["hybrid_rank_before_rerank"] = hybrid_rank
        result["reranker_score"] = float(score)
        reranked.append(result)

    reranked.sort(
        key=lambda x: x["reranker_score"],
        reverse=True,
    )

    reranked = reranked[:final_top_k]

    for rank, result in enumerate(reranked, start=1):
        result["reranked_rank"] = rank

    return reranked


def retrieve_context(
    connection,
    embedding_model,
    reranker,
    retrieval_query,
    publication_year_from=None,
    publication_year_to=None,
    author_contains=None,
):
    semantic = semantic_search(
        connection,
        embedding_model,
        retrieval_query,
        publication_year_from=publication_year_from,
        publication_year_to=publication_year_to,
        author_contains=author_contains,
    )

    lexical = lexical_search(
        connection,
        retrieval_query,
        publication_year_from=publication_year_from,
        publication_year_to=publication_year_to,
        author_contains=author_contains,
    )

    hybrid = reciprocal_rank_fusion(semantic, lexical)
    reranked = rerank_results(reranker, retrieval_query, hybrid)

    return {
        "semantic": semantic,
        "lexical": lexical,
        "hybrid": hybrid,
        "reranked": reranked,
    }


# ============================================================
# Grounded generation
# ============================================================

def build_context(reranked_results):
    blocks = []

    for index, item in enumerate(reranked_results, start=1):
        source_id = f"S{index}"

        block = f"""
[{source_id}]
Document ID: {item['document_id']}
Chunk ID: {item['chunk_id']}
Section: {item.get('section_title') or 'N/A'}
Publication year: {item.get('publication_year') or 'N/A'}
Authors: {item.get('authors') or 'N/A'}
Journal: {item.get('journal') or 'N/A'}

Text:
{item.get('content') or ''}
""".strip()

        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def generate_grounded_answer(
    client,
    original_question,
    retrieval_query,
    reranked_results,
):
    if not reranked_results:
        return (
            "No se encontró evidencia suficientemente relevante en el corpus "
            "científico recuperado para responder la pregunta."
        )

    context = build_context(reranked_results)

    system_prompt = """
You are the answer-generation component of a scientific literature RAG system.

Answer the user's question using ONLY the evidence contained in the retrieved
scientific sources.

Rules:
1. Use only information explicitly supported by the retrieved sources.
   Do not use your pre-trained knowledge to complete missing information.

2. Every scientific claim must be supported by at least one cited source,
   using labels such as [S1] or [S2][S4].

3. Before citing a source, ensure that the source actually supports the
   specific claim. Do not cite a source merely because it discusses the
   same disease or topic.

4. Do not introduce treatments, drugs, nutrients, mechanisms, statistics,
   clinical recommendations, approval status, or research directions that
   are not explicitly present in the retrieved sources.

5. If the evidence is incomplete, explicitly state that the retrieved
   literature is insufficient to answer that part of the question.

6. Do not imply that the retrieved evidence represents all existing
   scientific literature. Use wording such as "According to the retrieved
   literature" when appropriate.

7. Synthesize complementary evidence from multiple sources instead of
   summarizing each source independently.

8. Prioritize sources that directly answer the question, regardless of
   their position in the retrieved context.

9. Answer in the same language as the ORIGINAL QUESTION.

10. For medical or scientific questions, summarize the evidence rather
    than providing personalized medical advice.
""".strip()

    user_prompt = f"""
ORIGINAL QUESTION:
{original_question}

ENGLISH RETRIEVAL QUERY:
{retrieval_query}

RETRIEVED SCIENTIFIC SOURCES:
{context}

Answer the ORIGINAL QUESTION using only the retrieved evidence.
""".strip()

    response = client.chat.completions.create(
        model=get_groq_model(),
        temperature=0.1,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content.strip()


# ============================================================
# UI helpers
# ============================================================

def source_dataframe(results):
    rows = []

    for index, item in enumerate(results, start=1):
        rows.append(
            {
                "Source": f"S{index}",
                "Document ID": item["document_id"],
                "Chunk ID": item["chunk_id"],
                "Section": item.get("section_title"),
                "Year": item.get("publication_year"),
                "Journal": item.get("journal"),
                "Authors": ", ".join(item.get("authors") or [])
                    if isinstance(item.get("authors"), list)
                    else str(item.get("authors") or ""),
                "Reranker score": round(item.get("reranker_score", 0.0), 4),
                "RRF score": round(item.get("rrf_score", 0.0), 6),
                "Semantic rank": item.get("semantic_rank"),
                "Lexical rank": item.get("lexical_rank"),
            }
        )

    return pd.DataFrame(rows)


def load_evaluation_summary():
    candidates = [
        "evaluation_summary.csv",
        "evaluation_summary_current.csv",
    ]

    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)

    return None


# ============================================================
# App
# ============================================================
# ============================================================
# App
# ============================================================

# -----------------------------
# Sidebar navigation
# -----------------------------
with st.sidebar:
    st.markdown("## Scientific Literature Platform")

    if "page" not in st.session_state:
        st.session_state["page"] = "Chat"

    if st.button("💬  Chat", use_container_width=True):
        st.session_state["page"] = "Chat"

    if st.button("📑  Corpus Explorer", use_container_width=True):
        st.session_state["page"] = "Corpus Explorer"

    if st.button("📊  Evaluation", use_container_width=True):
        st.session_state["page"] = "Evaluation"

    st.divider()

    st.markdown("### System")
    st.markdown(
        f"""
**Embedding model:** `all-MiniLM-L6-v2`  
**Vector store:** `Supabase + pgvector`  
**Retrieval:** `Hybrid (Semantic + Lexical + RRF)`  
**Reranker:** `ms-marco-MiniLM-L-6-v2`  
**LLM provider:** `Groq`  
**LLM model:** `{get_groq_model()}`
"""
    )

    st.divider()
    st.caption("Developed by:")
    st.markdown("**Luis José Bolaños Berrocal**")


# -----------------------------
# Shared page title
# -----------------------------
st.title("🔬 Scientific Literature Platform")


# ============================================================
# CHAT PAGE
# ============================================================
if st.session_state["page"] == "Chat":
    st.header("💬 Ask Literature")
    st.caption("Ask questions against the indexed scientific literature.")

    with st.container(border=True):
        st.markdown("**Retrieval strategy**")
        st.text_input(
            "Retrieval strategy",
            value="Hybrid + Reranking",
            disabled=True,
            label_visibility="collapsed",
        )

    st.markdown("### Search filters")

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])

    enable_year_filter = filter_col1.checkbox("Filter by publication year")

    if enable_year_filter:
        year_from = filter_col1.number_input(
            "From year",
            min_value=1900,
            max_value=2100,
            value=2020,
            step=1,
        )
        year_to = filter_col2.number_input(
            "To year",
            min_value=1900,
            max_value=2100,
            value=2026,
            step=1,
        )
    else:
        year_from = None
        year_to = None

    author_filter = filter_col3.text_input(
        "Author contains",
        placeholder="e.g. Smith",
    ).strip()

    question = st.text_area(
        "Question",
        placeholder=(
            "Ask a scientific question in English, Spanish, Portuguese, "
            "or another language..."
        ),
        height=120,
    )

    run_search = st.button(
        "Search and answer",
        type="primary",
    )

    if run_search:
        if not question.strip():
            st.warning("Enter a question first.")
        elif (
            year_from is not None
            and year_to is not None
            and year_from > year_to
        ):
            st.warning("The starting year cannot be greater than the ending year.")
        else:
            try:
                embedding_model = load_embedding_model()
                reranker = load_reranker()
                groq_client = get_groq_client()

                with st.status("Running RAG pipeline...", expanded=True) as status:
                    st.write("Preparing the query for retrieval...")
                    retrieval_query = translate_query_for_retrieval(
                        groq_client,
                        question.strip(),
                    )

                    st.write(f"Retrieval query: `{retrieval_query}`")
                    st.write("Running semantic and lexical retrieval...")

                    connection = get_db_connection()

                    try:
                        retrieval = retrieve_context(
                            connection=connection,
                            embedding_model=embedding_model,
                            reranker=reranker,
                            retrieval_query=retrieval_query,
                            publication_year_from=year_from,
                            publication_year_to=year_to,
                            author_contains=author_filter or None,
                        )
                    finally:
                        connection.close()

                    final_sources = retrieval["reranked"]

                    st.write(
                        f"Reranked sources selected: {len(final_sources)}"
                    )

                    st.write("Generating grounded answer...")
                    answer = generate_grounded_answer(
                        client=groq_client,
                        original_question=question.strip(),
                        retrieval_query=retrieval_query,
                        reranked_results=final_sources,
                    )

                    status.update(
                        label="RAG pipeline completed",
                        state="complete",
                        expanded=False,
                    )

                st.session_state["last_question"] = question.strip()
                st.session_state["last_retrieval_query"] = retrieval_query
                st.session_state["last_answer"] = answer
                st.session_state["last_sources"] = final_sources
                st.session_state["last_retrieval"] = retrieval

            except Exception as exc:
                st.error(f"Error while running the RAG pipeline: {exc}")

    if "last_answer" in st.session_state:
        st.markdown("### Answer")
        st.markdown(st.session_state["last_answer"])

        with st.expander("Retrieval details"):
            st.write(
                "**Original question:**",
                st.session_state.get("last_question", ""),
            )
            st.write(
                "**English retrieval query:**",
                st.session_state.get("last_retrieval_query", ""),
            )

            retrieval = st.session_state.get("last_retrieval", {})

            if retrieval:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Semantic", len(retrieval.get("semantic", [])))
                c2.metric("Lexical", len(retrieval.get("lexical", [])))
                c3.metric("RRF", len(retrieval.get("hybrid", [])))
                c4.metric("Final context", len(retrieval.get("reranked", [])))

        sources = st.session_state.get("last_sources", [])
        if sources:
            st.markdown("### Sources")
            st.dataframe(
                source_dataframe(sources),
                use_container_width=True,
                hide_index=True,
            )

            for index, item in enumerate(sources, start=1):
                title = (
                    f"[S{index}] {item.get('section_title') or 'Untitled section'} "
                    f"— Document {item['document_id']}"
                )

                with st.expander(title):
                    st.write(f"**Year:** {item.get('publication_year') or 'N/A'}")
                    st.write(f"**Journal:** {item.get('journal') or 'N/A'}")
                    st.write(f"**Authors:** {item.get('authors') or 'N/A'}")
                    st.write(
                        f"**Reranker score:** "
                        f"{item.get('reranker_score', 0.0):.4f}"
                    )
                    st.write(item.get("content") or "")


# ============================================================
# CORPUS EXPLORER PAGE
# ============================================================
elif st.session_state["page"] == "Corpus Explorer":
    st.header("📑 Corpus Explorer")
    st.caption(
        "Explore documents and chunks stored in Supabase without running the RAG pipeline."
    )

    search_text = st.text_input(
        "Search in section title or chunk content",
        placeholder="e.g. insulin resistance, Gaucher, obesity...",
    ).strip()

    corpus_year = st.number_input(
        "Publication year (optional, 0 = all)",
        min_value=0,
        max_value=2100,
        value=0,
        step=1,
    )

    limit_rows = st.selectbox(
        "Rows to display",
        [25, 50, 100],
        index=0,
    )

    if st.button("Explore corpus", type="primary"):
        try:
            connection = get_db_connection()
            cursor = connection.cursor()

            sql = """
            SELECT
                c.chunk_id,
                c.document_id,
                c.section_title,
                d.publication_year,
                d.journal,
                d.authors,
                c.content
            FROM rag.document_chunks c
            JOIN rag.documents d
                ON d.document_id = c.document_id
            WHERE 1=1
            """

            params = []

            if search_text:
                sql += """
                AND (
                    COALESCE(c.section_title, '') ILIKE %s
                    OR COALESCE(c.content, '') ILIKE %s
                )
                """
                like_value = f"%{search_text}%"
                params.extend([like_value, like_value])

            if corpus_year:
                sql += "\nAND d.publication_year = %s"
                params.append(int(corpus_year))

            sql += "\nORDER BY d.publication_year DESC NULLS LAST, c.document_id, c.chunk_id"
            sql += "\nLIMIT %s"
            params.append(int(limit_rows))

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            columns = [
                "Chunk ID",
                "Document ID",
                "Section",
                "Year",
                "Journal",
                "Authors",
                "Content",
            ]

            corpus_df = pd.DataFrame(rows, columns=columns)

            cursor.close()
            connection.close()

            st.session_state["corpus_results"] = corpus_df

        except Exception as exc:
            st.error(f"Error while exploring the corpus: {exc}")

    corpus_df = st.session_state.get("corpus_results")

    if corpus_df is not None:
        st.dataframe(
            corpus_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Content": st.column_config.TextColumn(width="large"),
            },
        )


# ============================================================
# EVALUATION PAGE
# ============================================================
elif st.session_state["page"] == "Evaluation":
    st.header("📊 Retrieval Evaluation")
    st.caption(
        "Precomputed evaluation of semantic, lexical, hybrid RRF, "
        "and hybrid + reranking retrieval strategies."
    )

    evaluation_df = load_evaluation_summary()

    if evaluation_df is None:
        st.info(
            "No evaluation_summary.csv file was found. "
            "After rerunning Script 12 for the final corpus, place the "
            "aggregated evaluation CSV beside app.py."
        )
    else:
        st.dataframe(
            evaluation_df,
            use_container_width=True,
            hide_index=True,
        )

        metric_columns = [
            c for c in ["MRR", "P@5", "R@5", "nDCG@5", "R@10", "nDCG@10"]
            if c in evaluation_df.columns
        ]

        if "Method" in evaluation_df.columns and metric_columns:
            chart_df = evaluation_df.set_index("Method")[metric_columns]
            st.bar_chart(chart_df)

        st.info(
            "For the reranked system, @5 metrics are the primary comparison "
            "because the final RAG context contains five chunks."
        )

        with st.expander("How to interpret these metrics"):
            st.markdown(
                """
- **MRR** evaluates how quickly the first relevant result appears.
- **Precision@5** measures how many of the first five results are relevant.
- **Recall@5 / Recall@10** measure how much relevant evidence is retrieved.
- **nDCG** rewards systems that place highly relevant results near the top.

The evaluation is calculated offline by Script 12 and displayed here so the
web application does not rerun the full benchmark on every page load.
"""
            )
