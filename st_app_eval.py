import streamlit as st

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


with st.sidebar:
    # project_name = st.selectbox("Project Name", ["humaneval6", "weave-hooman1"])
    project_name = st.text_input("Project Name", "weave-hooman1")

    op = st_op_selectbox(
        project_name,
        "Op",
        sort_key=lambda o: (
            not "predict_and_score" in o.name,
            o.name,
        ),
    )
    if op is None:
        st.warning("no ops available")
        st.stop()

    calls2 = query.get_calls(project_name, op.name)
    calls = calls2.df

    compare_key = st_wv_column_selectbox(
        "A / B compare",
        calls2,
        op_types=("str",),
        sort_key=lambda col: (
            not col.name == "inputs.model",
            not col.name.startswith("inputs."),
            col.name,
        ),
    )
    if compare_key is None:
        st.warning("no compare keys available")
        st.stop()

    target_keys = st_wv_column_multiselect(
        "Target",
        calls2,
        # This was is_numeric_dtype instead of float64...
        op_types=("bool", "float64"),
        sort_key=lambda col: (
            not col.name.startswith("outputs."),
            col.name,
        ),
    )

    across_key = st_wv_column_selectbox(
        "Across",
        calls2,
        op_types=("str",),
        sort_key=lambda col: (
            not col.name == "inputs.example",
            not col.name.startswith("inputs."),
            col.name,
        ),
    )
    if across_key is None:
        st.warning("no across keys available")
        st.stop()

if len(target_keys) < 2:
    st.warning("Select at least two target keys")
    st.stop()

st.header(f"Comparing *{op.name}* calls by *{compare_key}*")

compare_val_stats_df, compare_vals = st_scatter_plot_mean(
    calls, compare_key, target_keys[1], target_keys[0]
)
if len(compare_vals) < 1:
    st.warning("Select one or more points on the chart above")
    st.stop()
if len(compare_vals) > 2:
    st.warning("We recommend selecting two points, for a more specific comparison")


if query.is_ref_series(compare_val_stats_df[compare_key]):
    expanded_df = query.resolve_refs(project_name, compare_val_stats_df[compare_key])
    compare_vals_df = expanded_df.loc[compare_vals]
    with st.expander(f"{len(compare_vals)} {compare_key} values"):
        if len(compare_vals_df) > 1:
            st_compare_dict(
                compare_vals_df.to_dict(orient="records"), compare_vals_df.index
            )
        else:
            st_dict(compare_vals_df.iloc[0])

##### Second level comparison by across_key #####

st.header(f"Split by *{across_key}*")

calls = calls[calls[compare_key].isin(compare_vals)]

if len(compare_vals) == 1:
    n_plot_cols = 2
    plot_cols = st.columns(n_plot_cols)
    for i, t in enumerate(target_keys):
        plot_col = plot_cols[i % n_plot_cols]
        with plot_col:
            st_barplot_plot_mean(calls, across_key, t)
elif len(compare_vals) == 2:
    n_plot_cols = 2
    plot_cols = st.columns(n_plot_cols)
    plot_selected_indexes = []
    for i, t in enumerate(target_keys):
        plot_col = plot_cols[i % n_plot_cols]
        with plot_col:
            selected = st_scatter_pivotxy_mean_histo(calls, across_key, compare_key, t)
            if len(selected):
                calls = calls[calls[across_key].isin(selected)]
else:
    _, across_vals = st_scatter_plot_mean(
        calls, across_key, target_keys[1], target_keys[0]
    )
    if len(across_vals):
        calls = calls[calls[across_key].isin(across_vals)]
    # for t in target_keys:
    #     t
    #     figs = st_n_histos(calls, across_key, compare_key, t)
    #     cols = st.columns(len(figs))
    #     for i, fig in enumerate(figs):
    #         with cols[i]:
    #             st.plotly_chart(fig, on_select="rerun")

    # cols = st.columns(len(compare_vals))
    # for i, c in enumerate(compare_vals):
    #     with cols[i]:
    #         st.write(st_safe_val(nice_ref(c)))

    # for c in compare_vals:
    #     st.write(c)
    #     cols = st.columns(len(target_keys))
    #     for i, t in enumerate(target_keys):
    #         with cols[i]:
    #             st_barplot_plot_mean(calls[calls[compare_key] == c], across_key, t)

    # for t in target_keys:

    # for t in target_keys:
    #     cols = st.columns(len(compare_vals))
    #     for i, c in enumerate(compare_vals):
    #         with cols[i]:
    #             st_barplot_plot_mean(calls[calls[compare_key] == c], across_key, t)
    #     # st_barplot_plot_mean(calls, across_key, t)

# Show a pivot across all target_keys
across_target_df = calls.pivot_table(
    index=across_key, columns=compare_key, values=target_keys, aggfunc="mean"
)
across_target_df.columns = across_target_df.columns.set_levels(
    across_target_df.columns.levels[1].map(nice_ref), level=1
)
across_target_df.columns = [
    ".".join((str(c) for c in col if c)) for col in across_target_df.columns
]
across_vals = across_target_df.index.to_series()
if query.is_ref_series(across_vals):
    across_vals = query.resolve_refs(project_name, across_vals)

row_selection = st.dataframe(
    across_target_df, on_select="rerun", selection_mode="single-row"
)

selected_rows = row_selection["selection"]["rows"]
if not selected_rows:
    st.warning("Select a row to view details.")
    st.stop()

across_selected_val = across_target_df.index[selected_rows[0]]
calls = calls[calls[across_key] == across_selected_val]
# across_val_resolved = query.resolve_refs(project_name, [across_selected_val]).iloc[0]
with st.expander(st_safe_val(nice_ref(across_selected_val))):
    st_dict(across_vals.loc[across_selected_val].to_dict())


st.subheader("calls")

grouped = calls.groupby(compare_key)
partitions = {name: group for name, group in grouped}
max_len = max(len(v) for v in partitions.values())

for i in range(max_len):
    st.subheader(f"{i}", divider=True)
    vals = [
        partitions[k].iloc[i].dropna() if i < len(partitions[k]) else {}
        for k in partitions
    ]
    if len(vals) > 1:
        st_multi_dict(vals, partitions.keys())
    else:
        st_dict(vals[0])
