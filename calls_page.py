import streamlit as st
import weave
import pandas as pd
import random
from weave.trace.refs import CallRef, parse_uri
import new_api

from pandas_util import pd_apply_and_insert

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


def call_refs(df):
    return [CallRef("none", "none", call_id).uri() for call_id in df["id"]]


def split_uri(uri):
    parts = uri.split("/")
    part_1 = "/".join(parts[:7])
    part_2 = "/".join(parts[7:])
    return pd.Series([part_1, part_2])


def pd_col_join(df, sep):
    columns = df.columns
    res = df[columns[0]]
    for i in range(1, len(columns)):
        res = res + sep + df[columns[i]]
    return res


def split_obj_ref(series):
    expanded = series.str.split("/", expand=True)
    name_version = expanded[6].str.split(":", expand=True)
    result = pd.DataFrame(
        {
            "entity": expanded[3],
            "project": expanded[4],
            "kind": expanded[5],
            "name": name_version[0],
            "version": name_version[1],
        }
    )
    if len(expanded.columns) > 7:
        result["path"] = pd_col_join(expanded.loc[:, expanded.columns > 6], "/")
    return result


def split_call_ref(series):
    expanded = series.str.split("/", expand=True)
    result = pd.DataFrame(
        {
            "entity": expanded[3],
            "project": expanded[4],
            "kind": expanded[5],
            "id": expanded[6],
        }
    )
    if len(expanded.columns) > 7:
        result["path"] = pd_col_join(expanded.loc[:, expanded.columns > 6], "/")
    return result


def split_call_ref_path(series):
    expanded = series.str.split("/", expand=True)
    result = pd.DataFrame(
        {
            "root": pd_col_join(expanded.loc[:, expanded.columns <= 6], "/"),
        }
    )
    if len(expanded.columns) > 7:
        result["path"] = pd_col_join(expanded.loc[:, expanded.columns > 6], "/")
    return result


@st.cache_resource
def init_weave():
    return weave.init_local_client()


client = init_weave()

op = st_op_selectbox(
    client,
    "Op",
    sort_key=lambda o: (
        not "predict_and_score" in o.name,
        o.name,
    ),
)
if op is None:
    st.warning("No ops available. Please type in a project name!")
    st.stop()

if "refesh_key" not in st.session_state:
    st.session_state["refesh_key"] = random.randrange(1000000000)

refresh_submitted = st.button("refresh")
if refresh_submitted:
    st.session_state["refesh_key"] += random.randrange(1000000000)

calls = query.get_calls(client, op.name, cache_key=st.session_state["refesh_key"])
calls_refs = call_refs(calls.df)
calls.df.index = calls_refs

calls_render = pd.concat(
    (
        calls.df.loc[:, calls.df.columns.str.startswith("inputs")],
        calls.df.loc[:, calls.df.columns.str.startswith("output")],
    ),
    axis=1,
)

all_calls = query.get_calls(client, None)

using_calls = query.get_calls(
    client, None, calls_refs, cache_key=st.session_state["refesh_key"]
)
using_calls = using_calls.df
if len(using_calls):
    using_calls = using_calls.explode("input_refs")

    using_calls = pd_apply_and_insert(using_calls, "input_refs", split_call_ref_path)
    using_calls = pd_apply_and_insert(using_calls, "op_name", split_obj_ref)
    using_calls_refs = call_refs(using_calls)
    using_calls.index = using_calls_refs
    using_calls["call_ref"] = using_calls.index
    op_counts = using_calls.groupby(["op_name.name", "op_name.version"]).size()

    # result = split_call_ref_path(using_calls["input_refs"])
    # result.columns = [".".join(("input_ref", col)) for col in result.columns]
    # using_calls = pd.concat([using_calls, result], axis=1)
    pivot = using_calls.pivot_table(
        index="input_refs.root", columns="op_name.name", aggfunc="last"
    )

    pivot_selected_columns = pd.concat(
        (
            # pivot.loc[:, pivot.columns.get_level_values(0) == "call_ref"],
            pivot.loc[:, pivot.columns.get_level_values(0).str.startswith("output")],
        ),
        axis=1,
    )
    pivot_selected_columns.columns = pivot_selected_columns.columns.swaplevel(0, 1)
    pivot_selected_columns.sort_index(axis=1, inplace=True)
    pivot_selected_columns.columns = [
        f"{col[0]}.{col[1]}" for col in pivot_selected_columns.columns
    ]

    calls_render = pd.concat(
        (calls_render, pivot_selected_columns),
        axis=1,
    )

# joined_render["fav"] = False
selected = st.dataframe(
    calls_render,
    hide_index=True,
    key="calls",
    selection_mode=("multi-row", "multi-column"),
    on_select="rerun",
    # num_rows="dynamic",
    # on_change=lambda arg: print(arg),
    # column_config={
    #     "fav": st.column_config.CheckboxColumn(
    #         "Your favorite?",
    #         help="Select your **favorite** widgets",
    #         default=False,
    #     )
    # },
)

# compute_df = pd.DataFrame(columns=calls_render.columns)
compute_df = pd.DataFrame(
    {col: pd.Series(dtype=dtype) for col, dtype in calls_render.dtypes.items()}
)

edited = st.data_editor(
    compute_df,
    hide_index=True,
    key="compute",
    num_rows="dynamic",
    # on_change=lambda arg: print(arg),
    # column_config={
    #     "fav": st.column_config.CheckboxColumn(
    #         "Your favorite?",
    #         help="Select your **favorite** widgets",
    #         default=False,
    #     )
    # },
)

# st.session_state

t = new_api.ComputeTable()
for col_name in compute_df.columns:
    if col_name.startswith("inputs"):
        t.add_input(col_name.split(".", 1)[1])
    else:
        if col_name == "output":
            op_name = op.name
        else:
            op_name = col_name.split(".", 1)[0]
        from weave.trace.refs import OpRef

        location = client._ref_uri(op_name, "latest", "obj")
        print("loc", location)
        weave_op = parse_uri(client._ref_uri(op_name, "latest", "obj")).get()
        ref = OpRef("None", "None", op_name, "latest")
        weave_op.ref = ref
        t.add_op(weave_op)

for row in st.session_state["compute"]["added_rows"]:
    row_to_add = {}
    for col_name in compute_df.columns:
        if col_name.startswith("inputs"):
            row_to_add[col_name.split(".", 1)[1]] = row[col_name]
    t.add_row(row_to_add)

run_button = st.button("Run")
status = t.status()
status_line = " ".join([f"{k} ({v['not_computed']} calls)" for k, v in status.items()])
to_compute_count = sum([v["not_computed"] for v in status.values()])
st.write(status_line)

compute_table_container = st.empty()
compute_table_df = t.dataframe()

if run_button:
    status_bar = st.progress(0)
    with compute_table_container:
        compute_table_df
    for i, delta in enumerate(t.execute()):
        status_bar.progress((i + 1) / to_compute_count)
        with compute_table_container:
            compute_table_df = t.dataframe()
            compute_table_df


rt = t.dataframe()
# status
# OK I'm hacking in cell selection by putting in editable checkbox columns
# and then I'll detect which cell is selected by looking for the checked one!


# expanded.columns = [
# expanded


# using_calls[["input_ref.root", "input_ref.path"]] = using_calls["input_refs"].apply(
#     split_uri
# )
# using_calls


# selection_state = st.dataframe(calls.df, selection_mode="single-row", on_select="rerun")
# selected_rows = selection_state["selection"]["rows"]
# if selected_rows:
#     selected_call = calls.df.loc[selected_rows[0]]
#     selected_call
#     selected_call_ref = CallRef("None", "None", selected_call.id).uri()
#     selected_call_ref

#     using_calls = query.get_calls(client, None, [selected_call_ref])
#     using_calls.df
