import streamlit as st
import weave
from weave.graph_client_context import set_graph_client
import asyncio
import plotly.express as px

import query
from st_components import (
    nice_ref,
    st_safe_val,
    st_op_selectbox,
    st_wv_column_selectbox,
    st_wv_column_multiselect,
    st_scatter_plot_mean,
    st_barplot_plot_mean,
    st_scatter_pivotxy_mean_histo,
    st_n_histos,
    st_dict,
    st_compare_dict,
    st_multi_dict,
)

st.set_page_config(layout="wide")

project_name = "weave-hooman1"
target_keys = [
    "output.model_output.answerable",
    "output.scores.llm_score.ai_response_correctness",
]


@st.cache_resource
def init_weave(project_name: str):
    return weave.init(project_name)


with st.spinner("weave.init"):
    wv = init_weave("weave-hooman1")

models = query.get_objs(wv, types="Model")

predscore_calls = query.get_calls(project_name, "Evaluation.predict_and_score")
predscore_calls_df = predscore_calls.df

predscore_calls_df = predscore_calls_df[
    predscore_calls_df["inputs.model"].isin(models.index)
]


with st.expander("Models"):
    compare_val_stats_df, compare_vals = st_scatter_plot_mean(
        predscore_calls_df, "inputs.model", target_keys[1], target_keys[0]
    )

    # models_display =
    model_stats_view = compare_val_stats_df
    if len(compare_vals) > 0:
        model_stats_view = model_stats_view.loc[compare_vals]

    selected = st.dataframe(
        model_stats_view,
        selection_mode="multi-row",
        on_select="rerun",
        hide_index=True,
    )
    selected_model_refs = model_stats_view.index[selected["selection"]["rows"]]
    # selected_row_indexes = selected["selection"]["rows"]

selected_calls = predscore_calls_df[
    predscore_calls_df["inputs.model"].isin(selected_model_refs)
]


with st.expander("Examples"):
    if len(compare_vals) == 1:
        example_pivot = selected_calls.pivot_table(
            index="inputs.example",
            columns="inputs.model",
            values=target_keys,
            aggfunc="mean",
        )
        example_pivot.columns = [".".join(col) for col in example_pivot.columns]
        n_plot_cols = 4
        plot_cols = st.columns(n_plot_cols)
        for i, t in enumerate(target_keys):
            plot_col = plot_cols[i % n_plot_cols]
            with plot_col:
                s = example_pivot[f"{t}.{compare_vals[0]}"]
                s_min, s_max = s.min(), s.max()

                result = st.slider(
                    t, min_value=s_min, max_value=s_max, value=(s_min, s_max)
                )
                fig = px.histogram(
                    s,
                    x=f"{t}.{compare_vals[0]}",
                    labels={f"{t}.{compare_vals[0]}": t},
                )
                st.plotly_chart(fig)

                example_pivot = example_pivot[
                    (example_pivot[f"{t}.{compare_vals[0]}"] >= result[0])
                    & (example_pivot[f"{t}.{compare_vals[0]}"] <= result[1])
                ]

        selected_ex = st.dataframe(
            example_pivot,
            selection_mode="multi-row",
            on_select="rerun",
            hide_index=True,
        )
        selected_ex_refs = example_pivot.index[selected_ex["selection"]["rows"]]
        selected_calls = selected_calls[
            selected_calls["inputs.example"].isin(selected_ex_refs)
        ]

model_details_df = query.resolve_refs(project_name, models.index.to_series())

if len(selected_model_refs):
    model_details_df = model_details_df.loc[selected_model_refs]
selected_evals = selected_calls["inputs.self"].value_counts()
selected_evals

st_compare_dict(model_details_df.to_dict(orient="records"), model_details_df.index)


selected_examples = selected_calls["inputs.example"].value_counts()
# selected_examples
# selected_calls
grouped = selected_calls.groupby(["inputs.example"])
for example, example_calls in selected_calls.groupby(["inputs.example"]):
    # st.write(example)
    # st.write(example_calls)
    # st.write('---')
    cols = st.columns([1] + [2] * len(selected_model_refs))
    cols_iter = iter(cols)
    col0 = next(cols_iter)
    with col0:
        st.write(f"**{example}**")
    for col, model_ref in zip(cols_iter, selected_model_refs):
        model_calls = example_calls[example_calls["inputs.model"] == model_ref]
        model_calls = model_calls.loc[:, model_calls.columns.str.startswith("output.")]
        with col:
            st_dict(model_calls.iloc[0].dropna())
        # st.write(model_calls)
# st.write(selected)
