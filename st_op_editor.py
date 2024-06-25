import streamlit as st
from typing import Callable
import inspect
import pandas as pd
import weave
from weave.graph_client_context import set_graph_client
from weave.trace_server.trace_server_interface import (
    ObjReadReq,
)
from pandas_util import *

import query
from weave.trace.refs import OpRef
from st_components import (
    st_op_selectbox,
    op_version_editor,
    st_project_picker,
    op_code_editor,
)

st.set_page_config(layout="wide")


def get_op_param_names(op: Callable):
    sig = inspect.signature(op)
    return list(sig.parameters.keys())


@st.cache_resource
def init_weave():
    return weave.init_local_client()


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


client = init_weave()

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
            if turn_on:
                st.session_state.version_checked = {
                    version.digest: digest == version.digest for version in versions
                }
            else:
                st.session_state.version_checked[digest] = False

        st.checkbox(
            f"v{version.version_index} ({version.call_count} calls)",
            value=st.session_state.version_checked.get(version.digest, False),
            on_change=update_checkbox,
            args=(version.digest,),
        )
first_version_that_is_checked = next(
    (v for v in versions if st.session_state.version_checked.get(v.digest, False)), None
)
if first_version_that_is_checked is None:
    st.warning("please select a version on the left")
    st.stop()
version_digest = first_version_that_is_checked.digest

op_fn = op_version_editor(client, sel_op.name, version_digest)

op_ref = OpRef(client.entity, client.project, sel_op.name, version_digest, [])

op_param_names = get_op_param_names(op_fn)

with st.form("run_op"):
    params = {k: st.text_input(k) for k in op_param_names}
    if st.form_submit_button("Run Op"):
        with set_graph_client(client):
            if op_fn is None:
                op_fn = op_ref.get()
            result = op_fn(**params)
            # st.rerun()

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
