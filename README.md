# TFM Scientific Literature RAG Platform

End-to-end scientific literature retrieval and RAG application for the Master's TFM. The repository contains the Streamlit application and the offline retrieval evaluation. Corpus ingestion/transformation and corpus embedding generation are executed by the separate Databricks/Cloud Run pipeline.

## Retrieval architecture

The application and evaluation share the retrieval implementation in `rag_core.py`:

- Semantic search: `sentence-transformers/all-MiniLM-L6-v2`, Top 20
- Lexical search: PostgreSQL full-text search, Top 20
- Hybrid fusion: Reciprocal Rank Fusion (RRF), `k=60`, Top 10
- Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Production RAG context: Top 5 reranked chunks sent to Groq
- Offline evaluation: Top 10 reranked chunks retained for fair metrics at k=10

The app also supports Vector, Lexical and Hybrid strategies for comparison. User queries can be translated to English with Groq before retrieval; the offline evaluation questions are kept in English so retrieval metrics do not depend on LLM translation.

## Repository structure

```text
.
├── app.py
├── rag_core.py
├── pages/
│   ├── chat.py
│   ├── corpus.py
│   └── evaluation.py
├── evaluation/
│   ├── retrieval_evaluation.py
│   ├── questions.csv
│   └── README.md
├── requirements.txt
└── .gitignore
```

## Run the application locally

```powershell
python -m streamlit run app.py
```

## Run the offline evaluation

From the repository root:

```powershell
python evaluation/retrieval_evaluation.py --mode pool
```

After manually judging every candidate (0–3):

```powershell
python evaluation/retrieval_evaluation.py --mode evaluate
```

The evaluator writes `evaluation_summary.csv` at the repository root so the Streamlit Evaluation page can display the final aggregate metrics.

## Secrets

Do not commit `.streamlit/secrets.toml` or database/API credentials. Use Streamlit secrets or environment variables.
