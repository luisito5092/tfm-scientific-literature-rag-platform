import os
import csv
import math
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import pg8000
from sentence_transformers import SentenceTransformer, CrossEncoder

# Reuse the exact retrieval implementation used by the Streamlit app.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_core import (
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    SEMANTIC_TOP_K,
    LEXICAL_TOP_K,
    HYBRID_TOP_K,
    RRF_K,
    semantic_search,
    lexical_search,
    reciprocal_rank_fusion,
    rerank_results,
)


# ============================================================
# 12 - Retrieval Evaluation
# ============================================================
#
# Compares:
#   1. Semantic
#   2. Lexical
#   3. Hybrid RRF
#   4. Hybrid RRF + Cross-Encoder reranker
#
# Retrieval functions are imported from rag_core.py so evaluation and app
# use the same implementation. Evaluation retains Top 10 after reranking;
# the production app intentionally sends only Top 5 chunks to Groq.
#
# Evaluation workflow:
#
#   A) POOL MODE
#      - Read questions.csv
#      - Run all retrieval methods
#      - Pool unique candidates
#      - Export candidate_pool.csv
#      - Manually assign relevance_grade:
#            0 = Not relevant
#            1 = Marginally relevant
#            2 = Relevant
#            3 = Highly relevant
#
#   B) EVALUATE MODE
#      - Read the manually judged candidate_pool.csv
#      - Re-run all retrieval methods
#      - Calculate:
#            Precision@k
#            Recall@k
#            MRR
#            nDCG@k
#
# IMPORTANT:
# Recall is relative to the judged pool, which is standard for pooled
# information-retrieval evaluation but is not guaranteed to represent
# every relevant chunk in the full corpus.
# ============================================================


# Evaluation depth is intentionally 10 even though the production app sends only Top 5 to Groq.
EVALUATION_RERANKED_TOP_K = 10
POOL_PER_METHOD = 10
EVAL_K_VALUES = [1, 3, 5, 10]


def get_connection():
    return pg8000.connect(
        host=os.environ["SUPABASE_HOST"],
        port=int(os.getenv("SUPABASE_PORT", "6543")),
        database=os.getenv("SUPABASE_DATABASE", "postgres"),
        user=os.environ["SUPABASE_USER"],
        password=os.environ["SUPABASE_PASSWORD"],
        ssl_context=True,
    )


def load_models():
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")

    print(f"Loading reranker: {RERANKER_MODEL_NAME}")
    reranker = CrossEncoder(RERANKER_MODEL_NAME)
    print("Reranker loaded.")

    return embedding_model, reranker


def run_all_methods(
    connection,
    embedding_model,
    reranker,
    question,
):
    """Run the four methods using rag_core.py, the same retrieval code as the app."""
    semantic = semantic_search(
        connection=connection,
        embedding_model=embedding_model,
        question=question,
        top_k=SEMANTIC_TOP_K,
    )

    lexical = lexical_search(
        connection=connection,
        question=question,
        top_k=LEXICAL_TOP_K,
    )

    hybrid = reciprocal_rank_fusion(
        semantic_results=semantic,
        lexical_results=lexical,
        rrf_k=RRF_K,
        final_top_k=HYBRID_TOP_K,
    )

    # Evaluation keeps all 10 reranked candidates so @10 metrics are valid.
    # The production app still uses only Top 5 as LLM context.
    reranked = rerank_results(
        reranker=reranker,
        question=question,
        hybrid_results=hybrid,
        final_top_k=EVALUATION_RERANKED_TOP_K,
    )

    return {
        "semantic": semantic,
        "lexical": lexical,
        "hybrid_rrf": hybrid,
        "hybrid_reranked": reranked,
    }


def read_questions(path):
    questions = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {"query_id", "question"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} must contain columns: query_id, question"
            )

        for row in reader:
            query_id = (row.get("query_id") or "").strip()
            question = (row.get("question") or "").strip()

            if query_id and question:
                questions.append({
                    "query_id": query_id,
                    "question": question,
                })

    return questions


def pool_candidates(
    questions,
    connection,
    embedding_model,
    reranker,
    output_path,
):
    rows = []

    for i, q in enumerate(questions, start=1):
        query_id = q["query_id"]
        question = q["question"]

        print(
            f"[{i}/{len(questions)}] Pooling candidates for "
            f"{query_id}: {question}"
        )

        methods = run_all_methods(
            connection,
            embedding_model,
            reranker,
            question,
        )

        pooled = {}

        for method_name, results in methods.items():
            for method_rank, item in enumerate(
                results[:POOL_PER_METHOD],
                start=1,
            ):
                chunk_id = int(item["chunk_id"])

                if chunk_id not in pooled:
                    pooled[chunk_id] = {
                        "query_id": query_id,
                        "question": question,
                        "chunk_id": chunk_id,
                        "document_id": item["document_id"],
                        "section_title": item.get("section_title") or "",
                        "publication_year": item.get("publication_year") or "",
                        "authors": str(item.get("authors") or ""),
                        "journal": item.get("journal") or "",
                        "content": item.get("content") or "",
                        "retrieved_by": set(),
                        "semantic_rank": "",
                        "lexical_rank": "",
                        "hybrid_rrf_rank": "",
                        "hybrid_reranked_rank": "",
                        "relevance_grade": "",
                        "judgment_notes": "",
                    }

                rec = pooled[chunk_id]
                rec["retrieved_by"].add(method_name)

                rank_field = f"{method_name}_rank"

                if method_name == "semantic":
                    rec[rank_field] = item.get(
                        "semantic_rank",
                        method_rank,
                    )
                elif method_name == "lexical":
                    rec[rank_field] = item.get(
                        "lexical_rank",
                        method_rank,
                    )
                elif method_name == "hybrid_rrf":
                    rec[rank_field] = method_rank
                elif method_name == "hybrid_reranked":
                    rec[rank_field] = item.get(
                        "reranked_rank",
                        method_rank,
                    )

        for rec in pooled.values():
            rec["retrieved_by"] = "|".join(
                sorted(rec["retrieved_by"])
            )
            rows.append(rec)

    fieldnames = [
        "query_id",
        "question",
        "chunk_id",
        "document_id",
        "section_title",
        "publication_year",
        "authors",
        "journal",
        "retrieved_by",
        "semantic_rank",
        "lexical_rank",
        "hybrid_rrf_rank",
        "hybrid_reranked_rank",
        "relevance_grade",
        "judgment_notes",
        "content",
    ]

    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Candidate pool created: {output_path}")
    print(f"Rows to judge manually: {len(rows)}")
    print()
    print("Assign relevance_grade:")
    print("  0 = No relevante")
    print("  1 = Poco relevante")
    print("  2 = Relevante")
    print("  3 = Muy relevante")


def read_qrels(path):
    qrels = defaultdict(dict)

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "query_id",
            "chunk_id",
            "relevance_grade",
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} must contain query_id, chunk_id, "
                "and relevance_grade"
            )

        missing = 0

        for row in reader:
            query_id = (row.get("query_id") or "").strip()
            chunk_id_raw = (row.get("chunk_id") or "").strip()
            grade_raw = (row.get("relevance_grade") or "").strip()

            if not query_id or not chunk_id_raw:
                continue

            if grade_raw == "":
                missing += 1
                continue

            grade = int(grade_raw)

            if grade not in {0, 1, 2, 3}:
                raise ValueError(
                    "relevance_grade must be one of: 0, 1, 2, 3"
                )

            qrels[query_id][int(chunk_id_raw)] = grade

    if missing:
        raise ValueError(
            f"{missing} rows still have blank relevance_grade. "
            "Judge all rows before evaluation."
        )

    return qrels


def precision_at_k(ranked_chunk_ids, relevance_map, k):
    retrieved = ranked_chunk_ids[:k]

    if k == 0:
        return 0.0

    relevant_count = sum(
        1
        for chunk_id in retrieved
        if relevance_map.get(chunk_id, 0) > 0
    )

    return relevant_count / k


def recall_at_k(ranked_chunk_ids, relevance_map, k):
    total_relevant = sum(
        1
        for grade in relevance_map.values()
        if grade > 0
    )

    if total_relevant == 0:
        return None

    retrieved_relevant = sum(
        1
        for chunk_id in ranked_chunk_ids[:k]
        if relevance_map.get(chunk_id, 0) > 0
    )

    return retrieved_relevant / total_relevant


def reciprocal_rank(ranked_chunk_ids, relevance_map):
    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        if relevance_map.get(chunk_id, 0) > 0:
            return 1.0 / rank

    return 0.0


def dcg_at_k(ranked_chunk_ids, relevance_map, k):
    dcg = 0.0

    for rank, chunk_id in enumerate(
        ranked_chunk_ids[:k],
        start=1,
    ):
        rel = relevance_map.get(chunk_id, 0)
        gain = (2 ** rel) - 1
        discount = math.log2(rank + 1)
        dcg += gain / discount

    return dcg


def ndcg_at_k(ranked_chunk_ids, relevance_map, k):
    actual_dcg = dcg_at_k(
        ranked_chunk_ids,
        relevance_map,
        k,
    )

    ideal_grades = sorted(
        relevance_map.values(),
        reverse=True,
    )[:k]

    ideal_dcg = 0.0

    for rank, rel in enumerate(ideal_grades, start=1):
        gain = (2 ** rel) - 1
        discount = math.log2(rank + 1)
        ideal_dcg += gain / discount

    if ideal_dcg == 0:
        return None

    return actual_dcg / ideal_dcg


def mean_ignore_none(values):
    valid = [v for v in values if v is not None]

    if not valid:
        return None

    return sum(valid) / len(valid)


def evaluate(
    questions,
    qrels,
    connection,
    embedding_model,
    reranker,
    output_path,
    summary_path,
):
    per_query_rows = []

    for i, q in enumerate(questions, start=1):
        query_id = q["query_id"]
        question = q["question"]

        if query_id not in qrels:
            print(
                f"Skipping {query_id}: no judged candidates."
            )
            continue

        print(
            f"[{i}/{len(questions)}] Evaluating "
            f"{query_id}: {question}"
        )

        methods = run_all_methods(
            connection,
            embedding_model,
            reranker,
            question,
        )

        relevance_map = qrels[query_id]

        for method_name, results in methods.items():
            ranked_ids = [
                int(item["chunk_id"])
                for item in results
            ]

            row = {
                "query_id": query_id,
                "question": question,
                "method": method_name,
                "MRR_component": reciprocal_rank(
                    ranked_ids,
                    relevance_map,
                ),
            }

            for k in EVAL_K_VALUES:
                row[f"Precision@{k}"] = precision_at_k(
                    ranked_ids,
                    relevance_map,
                    k,
                )
                row[f"Recall@{k}"] = recall_at_k(
                    ranked_ids,
                    relevance_map,
                    k,
                )
                row[f"nDCG@{k}"] = ndcg_at_k(
                    ranked_ids,
                    relevance_map,
                    k,
                )

            per_query_rows.append(row)

    fieldnames = [
        "query_id",
        "question",
        "method",
        "MRR_component",
    ]

    for k in EVAL_K_VALUES:
        fieldnames.extend([
            f"Precision@{k}",
            f"Recall@{k}",
            f"nDCG@{k}",
        ])

    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(per_query_rows)

    print()
    print("AGGREGATED RESULTS")
    print("=" * 110)

    by_method = defaultdict(list)

    for row in per_query_rows:
        by_method[row["method"]].append(row)

    summary = []

    for method_name, rows in by_method.items():
        result = {
            "method": method_name,
            "MRR": mean_ignore_none([
                r["MRR_component"]
                for r in rows
            ]),
        }

        for k in EVAL_K_VALUES:
            result[f"Precision@{k}"] = mean_ignore_none([
                r[f"Precision@{k}"]
                for r in rows
            ])
            result[f"Recall@{k}"] = mean_ignore_none([
                r[f"Recall@{k}"]
                for r in rows
            ])
            result[f"nDCG@{k}"] = mean_ignore_none([
                r[f"nDCG@{k}"]
                for r in rows
            ])

        summary.append(result)

    summary.sort(
        key=lambda x: (
            x.get("nDCG@5") is not None,
            x.get("nDCG@5") or -1,
        ),
        reverse=True,
    )

    # Streamlit-friendly aggregate schema.
    summary_rows = []
    for result in summary:
        row = {"Method": result["method"], "MRR": result["MRR"]}
        for k in EVAL_K_VALUES:
            row[f"P@{k}"] = result[f"Precision@{k}"]
            row[f"R@{k}"] = result[f"Recall@{k}"]
            row[f"nDCG@{k}"] = result[f"nDCG@{k}"]
        summary_rows.append(row)

    summary_fieldnames = ["Method", "MRR"]
    for k in EVAL_K_VALUES:
        summary_fieldnames.extend([f"P@{k}", f"R@{k}", f"nDCG@{k}"])

    with open(summary_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    header = (
        f"{'Method':<20}"
        f"{'MRR':>9}"
        f"{'P@5':>9}"
        f"{'R@5':>9}"
        f"{'nDCG@5':>11}"
        f"{'P@10':>9}"
        f"{'R@10':>9}"
        f"{'nDCG@10':>11}"
    )

    print(header)
    print("-" * len(header))

    def f(value):
        return "-" if value is None else f"{value:.4f}"

    for result in summary:
        print(
            f"{result['method']:<20}"
            f"{f(result['MRR']):>9}"
            f"{f(result['Precision@5']):>9}"
            f"{f(result['Recall@5']):>9}"
            f"{f(result['nDCG@5']):>11}"
            f"{f(result['Precision@10']):>9}"
            f"{f(result['Recall@10']):>9}"
            f"{f(result['nDCG@10']):>11}"
        )

    print()
    print(f"Per-query metrics saved to: {output_path}")
    print(f"Aggregate summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["pool", "evaluate"],
        required=True,
    )

    parser.add_argument(
        "--questions",
        default="evaluation/questions.csv",
    )

    parser.add_argument(
        "--pool-file",
        default="evaluation/candidate_pool.csv",
    )

    parser.add_argument(
        "--results-file",
        default="evaluation/evaluation_results.csv",
    )

    parser.add_argument(
        "--summary-file",
        default="evaluation_summary.csv",
        help="Aggregate metrics consumed by the Streamlit Evaluation page.",
    )

    args = parser.parse_args()

    questions = read_questions(args.questions)

    if not questions:
        raise ValueError("No questions found.")

    embedding_model, reranker = load_models()
    connection = get_connection()

    try:
        if args.mode == "pool":
            pool_candidates(
                questions=questions,
                connection=connection,
                embedding_model=embedding_model,
                reranker=reranker,
                output_path=args.pool_file,
            )

        elif args.mode == "evaluate":
            qrels = read_qrels(args.pool_file)

            evaluate(
                questions=questions,
                qrels=qrels,
                connection=connection,
                embedding_model=embedding_model,
                reranker=reranker,
                output_path=args.results_file,
                summary_path=args.summary_file,
            )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
