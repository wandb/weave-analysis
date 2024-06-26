import streamlit as st
from typing import Callable
import inspect
from weave.graph_client_context import set_graph_client
from pandas_util import *

import query
from weave.trace.refs import OpRef
from st_components import (
    st_op_selectbox,
    op_version_editor,
    st_project_picker,
    op_code_editor,
    st_safe_val,
)

st.set_page_config(layout="wide")


def get_op_param_names(op: Callable):
    sig = inspect.signature(op)
    return list(sig.parameters.keys())


DEFAULT_NEW_OP_CODE = """import weave
import json
import openai

@weave.op()
def describe(value):
    prompt = f'Describe this value: {value}'
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content"""


with st.sidebar:
    client = st_project_picker()
    sel_op = st_op_selectbox(client, "Op")

if sel_op is None:
    st.warning("No ops yet")
    op_fn = op_code_editor(client, DEFAULT_NEW_OP_CODE)
    op_param_names = get_op_param_names(op_fn)

    with st.form("run_op"):
        params = {k: st.text_input(k) for k in op_param_names}
        if st.form_submit_button("Run Op"):
            with set_graph_client(client):
                result = op_fn(**params)

    st.stop()

versions = query.get_op_versions(client, sel_op.name, True)
version_dicts = [
    {"version": "v" + str(v.version_index), "calls": v.call_count} for v in versions
]

if "version_checked" not in st.session_state:
    st.session_state.version_checked = {}
with st.sidebar:
    for version in versions:

        def update_checkbox(digest):
            turn_on = not st.session_state.version_checked.get(digest, False)
            st.session_state.version_checked[digest] = turn_on

        st.checkbox(
            f"v{version.version_index} ({version.call_count} calls)",
            value=st.session_state.version_checked.get(version.digest, False),
            on_change=update_checkbox,
            args=(version.digest,),
        )

checked_versions = [
    version
    for version in versions
    if st.session_state.version_checked.get(version.digest, False)
]

if len(checked_versions) == 0:
    st.warning("please select a version on the left")
    st.stop()

compare_columns = st.columns(len(checked_versions))

op_refs = []
op_fns = []

for col, version in zip(compare_columns, checked_versions):
    with col:
        st.write(st_safe_val(f"**{version.name}:v{version.version_index}**"))
        op_fn = op_version_editor(client, sel_op.name, version.digest)
        op_fns.append(op_fn)
        op_ref = OpRef(client.entity, client.project, sel_op.name, version.digest, [])
        op_refs.append(op_ref)

all_op_param_names = [get_op_param_names(op) for op in op_fns]

common_param_names = set(all_op_param_names[0])
for op_param_names in all_op_param_names[1:]:
    common_param_names = common_param_names.intersection(op_param_names)

for op_param_names in all_op_param_names:
    if set(op_param_names) != common_param_names:
        st.warning("Ops have different signatures")
        st.stop()


with st.form("run_op"):
    params = {k: st.text_input(k) for k in common_param_names}
    if st.form_submit_button("Run Op"):
        cols = st.columns(len(compare_columns))
        with set_graph_client(client):
            for col, op_ref, op_fn in zip(cols, op_refs, op_fns):
                if op_fn is None:
                    op_fn = op_ref.get()
                result = op_fn(**params)
                with col:
                    result
            # st.rerun()
cols = st.columns(len(compare_columns))
for col, op_ref in zip(cols, op_refs):
    with col:
        calls = query.get_calls(client, [op_ref.uri()])

        calls_view_df = calls.df[
            [col for col in calls.df.columns if col.startswith("started_at")]
            + [col for col in calls.df.columns if col.startswith("inputs")]
            + [col for col in calls.df.columns if col.startswith("output")]
        ].sort_values("started_at", ascending=False)

        st.dataframe(
            calls_view_df,
            column_config={
                "started_at": st.column_config.DatetimeColumn(
                    "Started", format="YYYY-MM-DD HH:mm:ss"
                )
            },
        )
