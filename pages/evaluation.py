import altair as alt
import pandas as pd
import streamlit as st

from rag_core import load_evaluation_summary

st.title("🔬 Scientific Literature Platform")
st.header("📊 Retrieval Evaluation")
st.caption(
    "Comparación de los mecanismos de búsqueda: Búsqueda léxica, semántica, híbrida e híbrida +  reranking"
)

evaluation_df = load_evaluation_summary()

if evaluation_df is None:
    st.info(
        "No se encontró el CSV evaluation_summary. Cree el evaluation_summary.csv "
    )
    st.stop()

# ---------------------------------------------------------------------
# Full metric table
# ---------------------------------------------------------------------
st.subheader("Métricas de Evaluación")

st.dataframe(
    evaluation_df,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "The table contains the complete retrieval evaluation. "
    "The charts below highlight candidate coverage first, followed by "
    "top-of-ranking quality."
)

# ---------------------------------------------------------------------
# Friendly method names
# ---------------------------------------------------------------------
method_labels = {
    "hybrid_reranked": "Híbrida + Reranking",
    "hybrid_rrf": "Híbrida RRF",
    "semantic": "Semántica",
    "lexical": "Léxica",
}

method_order = [
    "Híbrida + Reranking",
    "Híbrida RRF",
    "Semántica",
    "Léxica",
]

chart_df = evaluation_df.copy()

if "Method" in chart_df.columns:
    chart_df["Method label"] = chart_df["Method"].map(
        method_labels
    ).fillna(chart_df["Method"])
else:
    chart_df["Method label"] = chart_df.index.astype(str)

# ---------------------------------------------------------------------
# Chart 1: Recall @10 across retrieval strategies
# ---------------------------------------------------------------------
if "R@10" in chart_df.columns:
    st.subheader("Recall @10")
    st.caption(
        "Higher is better. This chart compares how much of the judged relevant "
        "evidence each strategy recovers within its first 10 ranked results."
    )

    recall_methods = chart_df[
        chart_df["Method"].isin(
            ["hybrid_reranked", "hybrid_rrf", "semantic", "lexical"]
        )
    ].copy()

    recall_order = ["Hybrid + Reranking", "Hybrid RRF", "Vector", "Lexical"]

    recall_chart = (
        alt.Chart(recall_methods)
        .mark_bar()
        .encode(
            x=alt.X(
                "Method label:N",
                sort=recall_order,
                title="Retrieval strategy",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "R@10:Q",
                title="Recall@10",
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("Method label:N", title="Retrieval"),
                alt.Tooltip("R@10:Q", title="Recall@10", format=".4f"),
            ],
        )
        .properties(height=340)
    )

    st.altair_chart(
        recall_chart,
        use_container_width=True,
    )

    st.info(
        "**How to read this chart:** compare the four strategies at the same "
        "cutoff. Offline evaluation retains 10 reranked candidates, making "
        "R@10 directly comparable across all methods. The production RAG still "
        "sends only the best five reranked chunks to Groq."
    )

# ---------------------------------------------------------------------
# Chart 2: Top-5 retrieval quality — grouped bars by retrieval strategy
# ---------------------------------------------------------------------
top5_metrics = [
    metric
    for metric in ["MRR", "P@5", "R@5", "nDCG@5"]
    if metric in chart_df.columns
]

if top5_metrics:
    st.subheader("Top-5 Retrieval Quality")
    st.caption(
        "Higher is better. Each retrieval strategy is represented on the x-axis, "
        "with four side-by-side bars comparing its Top-5 evaluation metrics."
    )

    quality_long = (
        chart_df[
            ["Method label"] + top5_metrics
        ]
        .melt(
            id_vars=["Method label"],
            value_vars=top5_metrics,
            var_name="Metric",
            value_name="Score",
        )
    )

    metric_order = ["MRR", "P@5", "R@5", "nDCG@5"]

    quality_chart = (
        alt.Chart(quality_long)
        .mark_bar()
        .encode(
            x=alt.X(
                "Method label:N",
                sort=method_order,
                title="Retrieval strategy",
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset=alt.XOffset(
                "Metric:N",
                sort=metric_order,
            ),
            y=alt.Y(
                "Score:Q",
                title="Metric score",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "Metric:N",
                sort=metric_order,
                title="Metric",
            ),
            tooltip=[
                alt.Tooltip("Method label:N", title="Retrieval"),
                alt.Tooltip("Metric:N", title="Metric"),
                alt.Tooltip("Score:Q", title="Score", format=".4f"),
            ],
        )
        .properties(height=380)
    )

    st.altair_chart(
        quality_chart,
        use_container_width=True,
    )

    st.info(
        "**How to read this chart:** Within each retrieval strategy, compare "
        "MRR, P@5, R@5, and nDCG@5. Across strategies, compare bars with the "
        "same metric. MRR rewards placing the first relevant result early; "
        "P@5 measures precision in the first five results; R@5 measures "
        "relevant-evidence coverage in those five positions; nDCG@5 rewards "
        "placing the most relevant evidence higher in the ranking."
    )

