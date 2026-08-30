# 14 — Streamlit RAG App v2.1

This is the UI layer of the TFM RAG platform.

## Main tabs

- Chat
- Corpus Explorer
- Evaluation

## Important

The UI is intentionally generic and is not hard-coded to Rare Genetic Diseases.

When the scientific corpus is changed, the application can be reused.
The corpus, embeddings, and evaluation results should be regenerated, but
the Streamlit UI and retrieval architecture do not need to be redesigned.

## Local execution

Install:

    python -m pip install -r requirements.txt

Create:

    .streamlit/secrets.toml

using `secrets.toml.example` as reference.

Then:

    streamlit run app.py

## Streamlit Community Cloud

Push the project to GitHub, then create an app in Streamlit Community Cloud
with:

    app.py

as the entry point.

Add all credentials in the application's Secrets configuration.

Never commit a real `secrets.toml` file.

## Evaluation tab

The included:

    evaluation_summary_current.csv

contains the current experimental results.

Once the final corpus/domain is selected and Script 12 is rerun, replace it
with the final evaluation output. The app first looks for:

    evaluation_summary.csv

and falls back to:

    evaluation_summary_current.csv

This means you can keep the current UI working while the corpus is rebuilt.

## Current architecture

    Databricks
      -> document ingestion / processing / chunking

    Supabase PostgreSQL + pgvector
      -> documents
      -> chunks
      -> permanent embeddings

    Streamlit
      -> query embedding
      -> semantic search
      -> lexical search
      -> RRF
      -> reranker

    Groq
      -> query translation
      -> grounded answer generation

## Notes

The final RAG context uses Top 5 reranked chunks.

The evaluation tab does not rerun Script 12 on every page load.
It only visualizes precomputed evaluation results.


## v2.2 visual layout

This version restores the original sidebar-oriented layout:

- Chat
- Corpus Explorer
- Evaluation

The Upload Document page was intentionally removed.

The System block in the sidebar reflects the current architecture rather than
the older Ollama/LangChain prototype.
