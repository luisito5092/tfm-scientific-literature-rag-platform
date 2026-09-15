import altair as alt
import pandas as pd
import streamlit as st

from rag_core import load_evaluation_summary


# ---------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------
st.title("🔬 Scientific Literature Platform")
st.header("📊 Retrieval Evaluation")

st.caption(
    "Comparación de los mecanismos de búsqueda léxica, semántica, "
    "híbrida RRF e híbrida con reranking."
)


# ---------------------------------------------------------------------
# Load evaluation results
# ---------------------------------------------------------------------
evaluation_df = load_evaluation_summary()

if evaluation_df is None:
    st.info(
        "No se encontró el archivo evaluation_summary.csv. "
        "Ejecute primero la evaluación del sistema."
    )
    st.stop()


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

display_df = evaluation_df.copy()

if "Method" in display_df.columns:
    display_df["Método"] = (
        display_df["Method"]
        .map(method_labels)
        .fillna(display_df["Method"])
    )
else:
    display_df["Método"] = display_df.index.astype(str)


# ---------------------------------------------------------------------
# Summary cards
# ---------------------------------------------------------------------
st.subheader("Resultados principales")

metric_columns = st.columns(3)

if "MRR" in display_df.columns:
    best_mrr_idx = display_df["MRR"].idxmax()
    best_mrr = display_df.loc[best_mrr_idx, "MRR"]
    best_mrr_method = display_df.loc[best_mrr_idx, "Método"]

    metric_columns[0].metric(
        label="Mejor MRR",
        value=f"{best_mrr:.4f}",
    )
    metric_columns[0].caption(best_mrr_method)

if "nDCG@5" in display_df.columns:
    best_ndcg5_idx = display_df["nDCG@5"].idxmax()
    best_ndcg5 = display_df.loc[best_ndcg5_idx, "nDCG@5"]
    best_ndcg5_method = display_df.loc[best_ndcg5_idx, "Método"]

    metric_columns[1].metric(
        label="Mejor nDCG@5",
        value=f"{best_ndcg5:.4f}",
    )
    metric_columns[1].caption(best_ndcg5_method)

if "nDCG@10" in display_df.columns:
    best_ndcg10_idx = display_df["nDCG@10"].idxmax()
    best_ndcg10 = display_df.loc[best_ndcg10_idx, "nDCG@10"]
    best_ndcg10_method = display_df.loc[best_ndcg10_idx, "Método"]

    metric_columns[2].metric(
        label="Mejor nDCG@10",
        value=f"{best_ndcg10:.4f}",
    )
    metric_columns[2].caption(best_ndcg10_method)


st.divider()


# ---------------------------------------------------------------------
# Compact evaluation table
# ---------------------------------------------------------------------
st.subheader("Comparación de estrategias")

preferred_metrics = [
    "MRR",
    "P@5",
    "R@5",
    "nDCG@5",
    "P@10",
    "R@10",
    "nDCG@10",
]

available_metrics = [
    metric
    for metric in preferred_metrics
    if metric in display_df.columns
]

table_df = display_df[
    ["Método"] + available_metrics
].copy()

# Preserve desired method order
table_df["Método"] = pd.Categorical(
    table_df["Método"],
    categories=method_order,
    ordered=True,
)

table_df = (
    table_df
    .sort_values("Método")
    .reset_index(drop=True)
)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        metric: st.column_config.NumberColumn(
            metric,
            format="%.4f",
        )
        for metric in available_metrics
    },
)

st.caption(
    "Agregación de los resultados obtenidos mediante el proceso de evaluación."
)


st.divider()


# ---------------------------------------------------------------------
# Main chart: ranking quality
# ---------------------------------------------------------------------
ndcg_metrics = [
    metric
    for metric in ["nDCG@5", "nDCG@10"]
    if metric in display_df.columns
]

if ndcg_metrics:
    st.subheader("Calidad del ordenamiento de resultados")

    st.caption(
        "nDCG evalúa no solo si se recupera evidencia relevante, sino también "
        "si los fragmentos con mayor relevancia aparecen en las primeras "
        "posiciones del ranking."
    )

    chart_source = display_df[
        ["Método"] + ndcg_metrics
    ].copy()

    chart_source["Método"] = pd.Categorical(
        chart_source["Método"],
        categories=method_order,
        ordered=True,
    )

    chart_source = (
        chart_source
        .sort_values("Método")
        .melt(
            id_vars=["Método"],
            value_vars=ndcg_metrics,
            var_name="Métrica",
            value_name="Score",
        )
    )

    ndcg_chart = (
        alt.Chart(chart_source)
        .mark_bar()
        .encode(
            x=alt.X(
                "Método:N",
                sort=method_order,
                title="Estrategia de recuperación",
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset=alt.XOffset(
                "Métrica:N",
                sort=["nDCG@5", "nDCG@10"],
            ),
            y=alt.Y(
                "Score:Q",
                title="nDCG",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "Métrica:N",
                sort=["nDCG@5", "nDCG@10"],
                title="Métrica",
            ),
            tooltip=[
                alt.Tooltip(
                    "Método:N",
                    title="Estrategia",
                ),
                alt.Tooltip(
                    "Métrica:N",
                    title="Métrica",
                ),
                alt.Tooltip(
                    "Score:Q",
                    title="Resultado",
                    format=".4f",
                ),
            ],
        )
        .properties(height=380)
    )

    st.altair_chart(
        ndcg_chart,
        use_container_width=True,
    )
