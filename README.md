# 14 - Streamlit RAG App v2.5

## Main change

The retrieval-strategy selector is now functional.

Default strategy:

- Hybrid + Reranking

Available strategies:

- Hybrid + Reranking
- Hybrid
- Vector
- Lexical

## Retrieval behavior

Vector:
- semantic embedding search
- final Top 5 sent to Groq

Lexical:
- PostgreSQL full-text search
- final Top 5 sent to Groq

Hybrid:
- Vector Top 20
- Lexical Top 20
- Reciprocal Rank Fusion
- final Top 5 sent to Groq

Hybrid + Reranking:
- Vector Top 20
- Lexical Top 20
- RRF Top 10
- CrossEncoder reranking
- final Top 5 sent to Groq

All strategies therefore provide the same maximum amount of context to the LLM.

## Filters

The Chat page keeps fixed metadata filters:

- From year
- To year
- Author

Both year dropdowns default to `All` and use the distinct years currently
available in Supabase.

## Navigation

The original `st.navigation` + `st.Page` layout is preserved.

## Run locally

```powershell
python -m streamlit run app.py
```


## v2.5.1 fix

Corrected the strategy wrapper so it calls the existing retrieval functions
using their actual parameter names:

- `semantic_search(..., question=..., top_k=...)`
- `lexical_search(..., question=..., top_k=...)`
- `reciprocal_rank_fusion(..., final_top_k=...)`
- `rerank_results(..., hybrid_results=..., final_top_k=...)`


## v2.6 - Corpus Explorer redesign

Corpus Explorer now:

- loads the corpus documents immediately
- shows Documents / Chunks / Embedded / Pending metrics
- displays document_id, title, source URL, publication year,
  embedding status and chunk count
- renders source URLs as clickable article links
- supports the same From year / To year pattern used in Chat
- defaults both year filters to All
- searches by title, source URL or document ID
- lets the user select any visible document and inspect all of its chunks
- shows chunk index, section title, section chunk index, character count,
  and complete chunk content

The v2.5.1 retrieval fix is preserved. The strategy wrapper continues to call:

- `semantic_search(..., question=..., top_k=...)`
- `lexical_search(..., question=..., top_k=...)`
- `reciprocal_rank_fusion(..., final_top_k=...)`
- `rerank_results(..., hybrid_results=..., final_top_k=...)`

The Chat strategies remain:

- Vector -> Top 5
- Lexical -> Top 5
- Hybrid -> Vector Top 20 + Lexical Top 20 -> RRF Top 5
- Hybrid + Reranking -> Vector Top 20 + Lexical Top 20
  -> RRF Top 10 -> CrossEncoder Top 5


## v2.7 - Strategy-aware Retrieved Evidence

Retrieved evidence now adapts to the active retrieval strategy:

- Vector: Semantic score + Semantic rank
- Lexical: Lexical score + Lexical rank
- Hybrid: RRF score + Semantic rank + Lexical rank
- Hybrid + Reranking: Reranker score + RRF score + Semantic rank + Lexical rank

Standalone vector and lexical searches now assign their own ranks directly.
Columns that do not apply are no longer shown as 0 or None.

The v2.5.1 fix remains preserved: semantic_search and lexical_search are
called with question=... and top_k=..., not query=... or limit=....


## v2.8 - Evaluation visualization redesign

Only the Evaluation page was changed.

The previous stacked bar chart was removed because adding MRR, Precision,
Recall and nDCG into one stacked total has no meaningful interpretation.

The Evaluation page now contains:

- the full metric table
- a grouped Top-5 Retrieval Quality chart using:
  - MRR
  - P@5
  - R@5
  - nDCG@5
- a separate Candidate Recall @10 chart
- an explanation of how to interpret each chart
- a warning that @10 metrics for Hybrid + Reranking are not directly
  comparable because the reranker returns only five final results
- metric definitions and a short explanation of how offline evaluation
  differs from the production RAG context size

All Chat, Corpus Explorer, retrieval strategy, ranking, and v2.5.1 fixes
remain unchanged from v2.7.


## v2.9 - Evaluation chart refinement

Only the Evaluation page changed functionally.

Chart order:
1. Candidate Recall @10 — Before Reranking
2. Top-5 Retrieval Quality

Candidate Recall @10:
- shows Hybrid RRF, Vector, and Lexical only
- excludes Hybrid + Reranking because the reranker returns only five final results
- emphasizes candidate coverage before CrossEncoder reranking

Top-5 Retrieval Quality:
- uses metrics as the chart categories:
  - MRR
  - P@5
  - R@5
  - nDCG@5
- compares retrieval methods side by side rather than stacking metric values
- avoids creating meaningless totals across unrelated IR metrics

All Chat, Corpus Explorer, strategy-aware evidence columns, ranking fixes,
and the v2.5.1 retrieval argument fix remain unchanged.


## v2.10 - Grouped Top-5 chart and cleaner navigation labels

Changes:

- Removed emoji characters from Streamlit page titles in `app.py`.
  Material icons remain unchanged.
- `Candidate Recall @10 — Before Reranking` remains the first chart.
- Replaced the Top-5 chart with an Altair grouped bar chart:
  - x-axis: retrieval strategy
  - y-axis: metric score from 0 to 1
  - four side-by-side bars per strategy:
    - MRR
    - P@5
    - R@5
    - nDCG@5
- The chart no longer stacks unrelated metrics.
- Added explicit `altair>=5,<6` dependency.

All previous retrieval, ranking, evidence-table, corpus, and v2.5.1 fixes
remain preserved.
