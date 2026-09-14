# Offline retrieval evaluation

This folder contains the offline evaluation workflow for the retrieval layer used by the Streamlit application. The evaluator imports the retrieval functions directly from `rag_core.py`, so the evaluated implementation is the same implementation used by the app.

The production app sends the final **Top 5** reranked chunks to Groq. Evaluation intentionally retains **Top 10** reranked chunks so Precision, Recall and nDCG at 10 are comparable across all four retrieval strategies.

## Workflow

Run commands from the repository root. Configure the Supabase connection with environment variables (`SUPABASE_HOST`, `SUPABASE_PORT`, `SUPABASE_DATABASE`, `SUPABASE_USER`, `SUPABASE_PASSWORD`). Do not commit credentials.

```powershell
python evaluation/retrieval_evaluation.py --mode pool
```

Manually assign `relevance_grade` in `evaluation/candidate_pool.csv`:

- 0 = not relevant
- 1 = marginally relevant
- 2 = relevant
- 3 = highly relevant

Then run:

```powershell
python evaluation/retrieval_evaluation.py --mode evaluate
```

This creates `evaluation/evaluation_results.csv` and the repository-root `evaluation_summary.csv`, which is consumed by the Streamlit Evaluation page.

`questions.csv` is kept from the previous experiment as a starting point. Candidate pools, judgments and metric outputs from the previous corpus are intentionally not included because the corpus is being rebuilt.
