from __future__ import annotations

import os

import pg8000.dbapi
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
EMBEDDING_DIMENSION = 384
ENCODING_BATCH_SIZE = int(os.getenv("ENCODING_BATCH_SIZE", "128"))
DB_WRITE_BATCH_SIZE = int(os.getenv("DB_WRITE_BATCH_SIZE", "256"))

SUPABASE_HOST = os.environ["SUPABASE_HOST"]
SUPABASE_PORT = int(os.getenv("SUPABASE_PORT", "6543"))
SUPABASE_DATABASE = os.getenv("SUPABASE_DATABASE", "postgres")
SUPABASE_USER = os.environ["SUPABASE_USER"]
SUPABASE_PASSWORD = os.environ["SUPABASE_PASSWORD"]

#Registro de logs
def log(message: str) -> None:
    print(message, flush=True)


#Conexion con Supabase
def get_connection():
    return pg8000.dbapi.connect(
        host=SUPABASE_HOST,
        port=SUPABASE_PORT,
        database=SUPABASE_DATABASE,
        user=SUPABASE_USER,
        password=SUPABASE_PASSWORD,
    )

#Ejecucion por batches
def batched(values, batch_size):
    for start in range(0, len(values), batch_size):
        yield values[start:start + batch_size]

#Creacion de formato de vector a almacenar
def vector_to_pgvector(vector) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in vector) + "]"

#Seleccionar chunks que requiren embedding, aquellos con estado pending o failed.
def fetch_chunks_to_embed(cursor):
    cursor.execute(
        """
        SELECT chunk_id, content
        FROM rag.document_chunks
        WHERE embedding_status IN ('pending', 'failed')
           OR embedding IS NULL
        ORDER BY chunk_id
        """
    )

    rows = cursor.fetchall()

    return [
        {"chunk_id": row[0], "content": row[1]}
        for row in rows
        if row[1]
    ]

#Bulk update de embeddings en Supabase
def bulk_update_embeddings(cursor, connection, rows):
    if not rows:
        return

    value_placeholders = ", ".join(
        ["(%s, %s, %s)" for _ in rows]
    )

    sql = f"""
        UPDATE rag.document_chunks AS target
        SET
            embedding = source.embedding_text::vector,
            embedding_model = source.embedding_model,
            embedding_status = 'completed',
            updated_at = NOW()
        FROM (
            VALUES {value_placeholders}
        ) AS source(
            chunk_id,
            embedding_text,
            embedding_model
        )
        WHERE target.chunk_id = source.chunk_id::bigint
    """

    params = []
    for chunk_id, embedding_text, embedding_model in rows:
        params.extend([
            chunk_id,
            embedding_text,
            embedding_model,
        ])

    try:
        cursor.execute(sql, tuple(params))
        connection.commit()
    except Exception:
        connection.rollback()
        raise

#Bulk update de embedding_status fallidos en Supabase
def bulk_mark_failed(cursor, connection, chunk_ids):
    if not chunk_ids:
        return

    placeholders = ", ".join(["%s" for _ in chunk_ids])

    sql = f"""
        UPDATE rag.document_chunks
        SET
            embedding_status = 'failed',
            updated_at = NOW()
        WHERE chunk_id IN ({placeholders})
    """

    try:
        cursor.execute(sql, tuple(chunk_ids))
        connection.commit()
    except Exception:
        connection.rollback()
        raise

#Bulk update del embedding_status de los documentos
def update_document_embedding_status(cursor, connection):
    """
    Derive rag.documents.embedding_status from its chunk states.

    Priority:
    1. failed     -> if any chunk failed
    2. pending    -> if any chunk remains pending or has no embedding
    3. completed  -> all chunks completed with embeddings
    """
    sql = """
        UPDATE rag.documents AS d
        SET
            embedding_status = status_summary.document_status,
            updated_at = NOW()
        FROM (
            SELECT
                document_id,
                CASE
                    WHEN COUNT(*) FILTER (
                        WHERE embedding_status = 'failed'
                    ) > 0
                    THEN 'failed'

                    WHEN COUNT(*) FILTER (
                        WHERE embedding_status = 'pending'
                           OR embedding IS NULL
                    ) > 0
                    THEN 'pending'

                    ELSE 'completed'
                END AS document_status
            FROM rag.document_chunks
            GROUP BY document_id
        ) AS status_summary
        WHERE d.document_id = status_summary.document_id
    """

    try:
        cursor.execute(sql)
        connection.commit()
    except Exception:
        connection.rollback()
        raise

#Consulta del embedding_status de los chunks
def get_embedding_status_summary(cursor):
    cursor.execute(
        """
        SELECT
            COUNT(*) AS total_chunks,
            COUNT(*) FILTER (
                WHERE embedding_status = 'completed'
                  AND embedding IS NOT NULL
            ) AS completed,
            COUNT(*) FILTER (
                WHERE embedding_status = 'pending'
            ) AS pending,
            COUNT(*) FILTER (
                WHERE embedding_status = 'failed'
            ) AS failed,
            COUNT(*) FILTER (
                WHERE embedding IS NULL
            ) AS missing_embeddings
        FROM rag.document_chunks
        """
    )

    row = cursor.fetchone()

    return {
        "total_chunks": row[0],
        "completed": row[1],
        "pending": row[2],
        "failed": row[3],
        "missing_embeddings": row[4],
    }

#Consulta del embedding_status de los documentos
def get_document_embedding_status_summary(cursor):
    cursor.execute(
        """
        SELECT
            COUNT(*) AS total_documents,
            COUNT(*) FILTER (
                WHERE embedding_status = 'completed'
            ) AS completed,
            COUNT(*) FILTER (
                WHERE embedding_status = 'pending'
            ) AS pending,
            COUNT(*) FILTER (
                WHERE embedding_status = 'failed'
            ) AS failed
        FROM rag.documents
        """
    )

    row = cursor.fetchone()

    return {
        "total_documents": row[0],
        "completed": row[1],
        "pending": row[2],
        "failed": row[3],
    }

#Ejecucion principal de las tareas del job
def main():
    log(f"Embedding model: {EMBEDDING_MODEL}")
    log(f"Embedding dimension: {EMBEDDING_DIMENSION}")
    log(f"Encoding batch size: {ENCODING_BATCH_SIZE}")
    log(f"DB write batch size: {DB_WRITE_BATCH_SIZE}")

    connection = get_connection()
    cursor = connection.cursor()

    try:
        chunks_to_embed = fetch_chunks_to_embed(cursor)
        total = len(chunks_to_embed)

        log(f"Chunks to embed/retry: {total}")

        if total == 0:
            log("No chunks require embeddings.")

            update_document_embedding_status(
                cursor,
                connection,
            )

            chunk_summary = get_embedding_status_summary(cursor)
            document_summary = get_document_embedding_status_summary(cursor)

            log(f"Chunk embedding status summary: {chunk_summary}")
            log(f"Document embedding status summary: {document_summary}")
            return

        log("Loading embedding model...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        log("Embedding model loaded.")

        processed = 0
        failed = 0

        for batch_number, batch in enumerate(
            batched(chunks_to_embed, ENCODING_BATCH_SIZE),
            start=1,
        ):
            texts = [row["content"] for row in batch]
            chunk_ids = [row["chunk_id"] for row in batch]

            try:
                embeddings = model.encode(
                    texts,
                    batch_size=ENCODING_BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                if (
                    embeddings.ndim != 2
                    or embeddings.shape[1] != EMBEDDING_DIMENSION
                ):
                    raise RuntimeError(
                        f"Unexpected embedding shape: {embeddings.shape}"
                    )

                update_rows = [
                    (
                        chunk_id,
                        vector_to_pgvector(embedding),
                        EMBEDDING_MODEL,
                    )
                    for chunk_id, embedding in zip(
                        chunk_ids,
                        embeddings,
                    )
                ]

                for write_batch in batched(
                    update_rows,
                    DB_WRITE_BATCH_SIZE,
                ):
                    bulk_update_embeddings(
                        cursor,
                        connection,
                        write_batch,
                    )

                processed += len(batch)

                log(
                    f"Batch {batch_number}: "
                    f"processed {processed}/{total}"
                )

            except Exception as error:
                failed += len(batch)

                log(
                    f"Batch {batch_number} FAILED: "
                    f"{type(error).__name__}: {error}"
                )

                bulk_mark_failed(
                    cursor,
                    connection,
                    chunk_ids,
                )

        # Synchronize document-level embedding status from chunk-level truth.
        update_document_embedding_status(
            cursor,
            connection,
        )

        chunk_summary = get_embedding_status_summary(cursor)
        document_summary = get_document_embedding_status_summary(cursor)

        log("Embedding job finished.")
        log(f"Processed successfully in this run: {processed}")
        log(f"Failed in this run: {failed}")
        log(f"Chunk embedding status summary: {chunk_summary}")
        log(f"Document embedding status summary: {document_summary}")

        if failed > 0:
            raise RuntimeError(
                f"{failed} chunks failed in this execution."
            )

    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    main()
