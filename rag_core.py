import os
from typing import Optional

import pandas as pd
import pg8000
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# ==================== Streamlit RAG App =====================
# ============================================================

# ============================================================
# Variables
# ============================================================
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

    results = [dict(zip(columns, row)) for row in rows]

    for rank, item in enumerate(results, start=1):
        item["semantic_rank"] = rank

    return results


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

    results = [dict(zip(columns, row)) for row in rows]

    for rank, item in enumerate(results, start=1):
        item["lexical_rank"] = rank

    return results


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


def source_dataframe(results, strategy=None):
    """Build a strategy-aware retrieval evidence dataframe."""
    rows = []

    for index, item in enumerate(results, start=1):
        row = {
            "Source": f"S{index}",
            "Document ID": item["document_id"],
            "Chunk ID": item["chunk_id"],
            "Section": item.get("section_title"),
            "Year": item.get("publication_year"),
            "Journal": item.get("journal"),
            "Authors": (
                ", ".join(item.get("authors") or [])
                if isinstance(item.get("authors"), list)
                else str(item.get("authors") or "")
            ),
        }

        if strategy == "Vector":
            score = item.get("semantic_score")
            row["Semantic score"] = round(float(score), 4) if score is not None else None
            row["Semantic rank"] = item.get("semantic_rank")

        elif strategy == "Lexical":
            score = item.get("lexical_score")
            row["Lexical score"] = round(float(score), 4) if score is not None else None
            row["Lexical rank"] = item.get("lexical_rank")

        elif strategy == "Hybrid":
            score = item.get("rrf_score")
            row["RRF score"] = round(float(score), 6) if score is not None else None
            row["Semantic rank"] = item.get("semantic_rank")
            row["Lexical rank"] = item.get("lexical_rank")

        elif strategy == "Hybrid + Reranking":
            reranker_score = item.get("reranker_score")
            rrf_score = item.get("rrf_score")
            row["Reranker score"] = (
                round(float(reranker_score), 4)
                if reranker_score is not None else None
            )
            row["RRF score"] = (
                round(float(rrf_score), 6)
                if rrf_score is not None else None
            )
            row["Semantic rank"] = item.get("semantic_rank")
            row["Lexical rank"] = item.get("lexical_rank")

        rows.append(row)

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


@st.cache_data(ttl=600)
def get_available_publication_years():
    """Return distinct publication years actually present in rag.documents."""
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT DISTINCT publication_year
            FROM rag.documents
            WHERE publication_year IS NOT NULL
            ORDER BY publication_year ASC
            """
        )
        return [int(row[0]) for row in cursor.fetchall()]
    finally:
        cursor.close()
        connection.close()




@st.cache_data(ttl=120)
def get_corpus_stats():
    """
    Return corpus-level counts:
      documents, chunks, embedded chunks, pending chunks.
    """
    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM rag.documents) AS documents,
                COUNT(*) AS chunks,
                COUNT(*) FILTER (
                    WHERE embedding IS NOT NULL
                      AND embedding_status = 'completed'
                ) AS embedded,
                COUNT(*) FILTER (
                    WHERE embedding IS NULL
                       OR COALESCE(embedding_status, '') <> 'completed'
                ) AS pending
            FROM rag.document_chunks
            """
        )
        row = cursor.fetchone()
    finally:
        cursor.close()
        connection.close()

    return {
        "documents": int(row[0] or 0),
        "chunks": int(row[1] or 0),
        "embedded": int(row[2] or 0),
        "pending": int(row[3] or 0),
    }


@st.cache_data(ttl=120)
def get_corpus_documents(
    search_text="",
    publication_year_from=None,
    publication_year_to=None,
):
    """
    Return the documents that form the corpus, together with their chunk count.
    The query runs on page load and can be narrowed by text and year range.
    """
    connection = get_db_connection()
    cursor = connection.cursor()

    sql = """
    SELECT
        d.document_id,
        d.title,
        d.source_url,
        d.publication_year,
        d.embedding_status,
        COUNT(c.chunk_id) AS chunk_count
    FROM rag.documents d
    LEFT JOIN rag.document_chunks c
        ON c.document_id = d.document_id
    WHERE 1=1
    """

    params = []

    if search_text:
        sql += """
        AND (
            COALESCE(d.title, '') ILIKE %s
            OR COALESCE(d.source_url, '') ILIKE %s
            OR CAST(d.document_id AS text) ILIKE %s
        )
        """
        like_value = f"%{search_text}%"
        params.extend([like_value, like_value, like_value])

    if publication_year_from is not None:
        sql += "\nAND d.publication_year >= %s"
        params.append(int(publication_year_from))

    if publication_year_to is not None:
        sql += "\nAND d.publication_year <= %s"
        params.append(int(publication_year_to))

    sql += """
    GROUP BY
        d.document_id,
        d.title,
        d.source_url,
        d.publication_year,
        d.embedding_status
    ORDER BY
        d.document_id
    """

    try:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()

    columns = [
        "document_id",
        "title",
        "source_url",
        "publication_year",
        "embedding_status",
        "chunk_count",
    ]

    return [dict(zip(columns, row)) for row in rows]


@st.cache_data(ttl=120)
def get_document_chunks(document_id):
    """
    Return all chunks for a selected document in their original chunk order.
    """
    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            SELECT
                c.chunk_id,
                c.chunk_index,
                c.section_chunk_index,
                c.section_title,
                c.content
            FROM rag.document_chunks c
            WHERE c.document_id = %s
            ORDER BY
                c.chunk_index NULLS LAST,
                c.chunk_id
            """,
            [int(document_id)],
        )
        rows = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()

    columns = [
        "chunk_id",
        "chunk_index",
        "section_chunk_index",
        "section_title",
        "content",
    ]

    return [dict(zip(columns, row)) for row in rows]


def retrieve_by_strategy(
    connection,
    embedding_model,
    reranker,
    retrieval_query,
    strategy,
    publication_year_from=None,
    publication_year_to=None,
    author_contains=None,
):
    """
    Execute the selected retrieval strategy and always return at most 5
    final chunks for the LLM.

    Strategies:
      - Vector: semantic Top 5
      - Lexical: lexical Top 5
      - Hybrid: semantic Top 20 + lexical Top 20 -> RRF -> Top 5
      - Hybrid + Reranking: semantic Top 20 + lexical Top 20
        -> RRF Top 10 -> CrossEncoder -> Top 5
    """
    valid = {
        "Vector",
        "Lexical",
        "Hybrid",
        "Hybrid + Reranking",
    }

    if strategy not in valid:
        raise ValueError(f"Unsupported retrieval strategy: {strategy}")

    common_filters = dict(
        publication_year_from=publication_year_from,
        publication_year_to=publication_year_to,
        author_contains=author_contains,
    )

    if strategy == "Vector":
        semantic = semantic_search(
            connection=connection,
            embedding_model=embedding_model,
            question=retrieval_query,
            top_k=5,
            **common_filters,
        )

        final = semantic[:5]

        return {
            "strategy": strategy,
            "semantic": semantic,
            "lexical": [],
            "hybrid": [],
            "reranked": [],
            "final": final,
        }

    if strategy == "Lexical":
        lexical = lexical_search(
            connection=connection,
            question=retrieval_query,
            top_k=5,
            **common_filters,
        )

        final = lexical[:5]

        return {
            "strategy": strategy,
            "semantic": [],
            "lexical": lexical,
            "hybrid": [],
            "reranked": [],
            "final": final,
        }

    semantic = semantic_search(
        connection=connection,
        embedding_model=embedding_model,
        question=retrieval_query,
        top_k=20,
        **common_filters,
    )

    lexical = lexical_search(
        connection=connection,
        question=retrieval_query,
        top_k=20,
        **common_filters,
    )

    if strategy == "Hybrid":
        hybrid = reciprocal_rank_fusion(
            semantic_results=semantic,
            lexical_results=lexical,
            final_top_k=5,
        )

        return {
            "strategy": strategy,
            "semantic": semantic,
            "lexical": lexical,
            "hybrid": hybrid,
            "reranked": [],
            "final": hybrid[:5],
        }

    hybrid = reciprocal_rank_fusion(
        semantic_results=semantic,
        lexical_results=lexical,
        final_top_k=10,
    )

    reranked = rerank_results(
        reranker=reranker,
        question=retrieval_query,
        hybrid_results=hybrid,
        final_top_k=5,
    )

    return {
        "strategy": strategy,
        "semantic": semantic,
        "lexical": lexical,
        "hybrid": hybrid,
        "reranked": reranked,
        "final": reranked[:5],
    }

def retrieval_strategy_description(strategy):
    descriptions = {
        "Vector": "Semantic embedding search → Top 5",
        "Lexical": "PostgreSQL full-text search → Top 5",
        "Hybrid": "Vector Top 20 + Lexical Top 20 → RRF → Top 5",
        "Hybrid + Reranking": (
            "Vector Top 20 + Lexical Top 20 → RRF Top 10 "
            "→ CrossEncoder → Top 5"
        ),
    }
    return descriptions[strategy]
