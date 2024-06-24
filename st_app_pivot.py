import streamlit as st
from typing import Callable
import inspect
import json
import pandas as pd
import weave
from weave.graph_client_context import set_graph_client
from code_editor import code_editor
import openai
from weave_api_next import weave_client_calls
from streamlit_profiler import Profiler
from weave.trace.custom_objs import MemTraceFilesArtifact
from weave.trace import op_type
from weave.trace_server.trace_server_interface import (
    FileContentReadReq,
    ObjReadReq,
)
from pandas_util import *

import query
import new_api
from weave.trace.refs import OpRef
from st_components import st_safe_val

st.set_page_config(layout="wide")


@st.cache_resource
def init_weave():
    return weave.init_local_client()


def get_op_param_names(op: Callable):
    sig = inspect.signature(op)
    return list(sig.parameters.keys())


# with Profiler():
client = init_weave()

##### Dataset Uploader

uploaded_file = st.file_uploader("Choose a file", type=["jsonl"])
if uploaded_file:
    data = [
        json.loads(line)
        for line in uploaded_file.getvalue().decode().split("\n")
        if line
    ]
    uploaded_file.name
    name = uploaded_file.name.rsplit(".", 1)[0]
    df = pd.DataFrame(data)
    with set_graph_client(client):
        weave.publish(weave.Dataset(name=name, rows=data))

##### Dataset selector and loading.

objs = client.objects()
datasets = [f"{o.val.get('name')}:{o.digest}" for o in objs]
col_a, col_b = st.columns(2)
with col_a:
    dataset_name_v = st.selectbox("Dataset", datasets)
with col_b:
    limit = st.number_input("Limit", min_value=0, value=10)
if not dataset_name_v:
    st.stop()
with set_graph_client(client):
    dataset = weave.ref(dataset_name_v).get()

    ct = new_api.ComputeTable(dataset.rows, limit=limit)

    ds_rows = list(dataset.rows)
    ds_rows_refs = [row.ref.uri() for row in ds_rows]
    ds_df = pd.DataFrame(ds_rows, index=ds_rows_refs)


##### Infer compute table from using_calls

with set_graph_client(client):
    all_using_calls = query.get_calls(client, None, ds_rows_refs)
last_versions = {}
name_to_versions = {}

if len(all_using_calls.df):
    last_versions = all_using_calls.df.groupby("op_name.name")["op_name.version"].last()

    df_unique_ordered = all_using_calls.df.drop_duplicates(
        subset=["op_name.name", "op_name.version"]
    )
    name_to_versions = (
        df_unique_ordered.groupby("op_name.name")["op_name.version"]
        .apply(list)
        .to_dict()
    )

failed_ops = []

for op_name, last_version in last_versions.items():
    with set_graph_client(client):
        op_ref = OpRef("none", "none", op_name, last_version, [])
        op = op_ref.get()
    try:
        ct.add_op(op)
    except ValueError:
        failed_ops.append(op_name)
really_failed_ops = []
for op_name in failed_ops:
    with set_graph_client(client):
        op_ref = OpRef("none", "none", op_name, last_versions[op_name], [])
        op = op_ref.get()
    try:
        ct.add_op(op)
    except ValueError:
        pass

if really_failed_ops:
    st.warning("Failed to add ops: " + ", ".join(really_failed_ops))

with set_graph_client(client):
    ct.fill_from_cache()

##### Render compute table.

ct_df = ct.dataframe()
sel = st.dataframe(
    ct_df,
    selection_mode="single-row",
    on_select="rerun",
    hide_index=True,
    use_container_width=True,
)


ct_status = ct.status()
total_to_run = sum(op_status["not_computed"] for op_status in ct_status.values())

if st.button(f"Fill {total_to_run} blanks"):
    op_strings = [
        f'{op_name} ({op_status["not_computed"]})'
        for op_name, op_status in ct_status.items()
    ]
    op_string = ", ".join(op_strings)
    my_bar = st.progress(0, text=op_string)
    with set_graph_client(client):
        for i, cell_result in enumerate(ct.execute()):
            op_strings = [
                f'{op_name} ({op_status["not_computed"]})'
                for op_name, op_status in ct_status.items()
            ]
            op_string = ", ".join(op_strings)
            my_bar.progress((i + 1) / total_to_run, text=op_string)
    st.rerun()


##### Individual row view.
# This shows us the selected row, and also
# acts as a column editor for the ComputeTable

sel_rows = sel["selection"]["rows"]
col_sizes = [1, 5]
if sel_rows:
    row = ds_df.iloc[sel_rows[0]]
    st.write(st_safe_val(row.name))
    avail_vars = dict(row.items())

    ##### Dataset values for row.
    for k, v in row.items():
        key_col, val_col = st.columns(col_sizes)
        with key_col:
            st.write(f"**{k}**")
        with val_col:
            st.json(v, expanded=False)

    # Redundant using calls fetch to above.
    with set_graph_client(client):
        using_calls = query.get_calls(client, None, [row.name])
    using_calls_by_name = None
    if len(using_calls.df):
        using_calls_by_name = using_calls.df.groupby("op_name.name")

    # Render compute columns
    for col_name, col in ct.columns.items():
        computed_val = ct.rows[sel_rows[0]][col_name]
        if not isinstance(col, new_api.OpColumn):
            continue
        op_name = col.op.ref.name
        key_col, val_col = st.columns(col_sizes)

        with key_col:
            st.write(st_safe_val(f"**{col_name}**"))
        with val_col:
            try:
                str_computed_val = json.dumps(computed_val)
            except TypeError:
                str_computed_val = str(computed_val)
            if len(str_computed_val) > 100:
                str_computed_val = str_computed_val[:100] + "..."
            with st.expander(str_computed_val):
                value = None
                versions = list(reversed(name_to_versions[op_name]))
                version_index = versions.index(col.op.ref.digest)
                version = st.selectbox(
                    "version",
                    versions,
                    index=version_index,
                )

                ##### Load op code
                fn = None
                with set_graph_client(client):
                    server_read = client.server.obj_read(
                        ObjReadReq(
                            project_id="none/none", object_id=op_name, digest=version
                        )
                    )
                    code_file_digest = server_read.obj.val["files"]["obj.py"]
                    code_file_contents = client.server.file_content_read(
                        FileContentReadReq(
                            project_id="none/none", digest=code_file_digest
                        )
                    ).content.decode()
                    response_dict = code_editor(
                        code_file_contents, key=f"code_editor_{op_name}_{version}"
                    )
                    if response_dict["text"]:
                        exec_locals = {}
                        exec(response_dict["text"], globals(), exec_locals)
                        art = MemTraceFilesArtifact(
                            {"obj.py": response_dict["text"].encode()}
                        )
                        fn = op_type.load_instance(art, "obj")

                if st.button("Run Op", key=f"run_op_{col_name}"):
                    op_ref = OpRef("none", "none", op_name, version, [])
                    with set_graph_client(client):
                        if fn is None:
                            fn = op_ref.get()
                        op_param_names = get_op_param_names(fn)
                        params = {k: avail_vars[k] for k in op_param_names}
                        result = fn(**params)
                    st.rerun()
                if (
                    using_calls_by_name is not None
                    and op_name in using_calls_by_name.groups
                ):
                    op_calls_df = using_calls_by_name.get_group(op_name)

                    st.write("prior calls")
                    op_version_calls = op_calls_df[
                        op_calls_df["op_name.version"] == version
                    ]
                    if len(op_version_calls) > 0:
                        value = get_unflat_value(
                            op_version_calls.iloc[0].dropna(), "output"
                        )
                        op_sel = st.dataframe(
                            op_version_calls.filter(regex=f"^(inputs|output)"),
                            use_container_width=True,
                            selection_mode="single-row",
                            on_select="rerun",
                        )
                        op_sel_rows = op_sel["selection"]["rows"]
                        if op_sel_rows:
                            call = op_version_calls.iloc[op_sel_rows[0]]
                            populated_columns = call.dropna()
                            if "output" in populated_columns:
                                value = call["output"]
                            else:
                                value = call.dropna().filter(regex=f"^output").to_dict()
                                # remove 'output.' prefix
                                value = {k[7:]: v for k, v in value.items()}
                if value is not None:
                    avail_vars[col_name] = value
                    value
    key_col, val_col = st.columns(col_sizes)
    with key_col:
        st.write("**New Op**")
    with val_col:
        st.write("available vars", avail_vars.keys())
        k0 = list(row.keys())[0]
        op_code = f"""import weave
import json
import openai

@weave.op()
def explain({k0}):
    prompt = f'Explain this: {{json.dumps({k0})}}'
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{{"role": "user", "content": prompt}}],
    )
    return response.choices[0].message.content"""
        initial_op_name = "explain"
        key_col, val_col = st.columns((1, 5))

        response_dict = code_editor(op_code)
        exec_locals = {}

        if response_dict["text"]:
            op_code = response_dict["text"]

        exec(op_code, globals(), exec_locals)

        art = MemTraceFilesArtifact({"obj.py": op_code.encode()})
        fn = op_type.load_instance(art, "obj")
        # fn = weave.op()(exec_locals[list(exec_locals.keys())[-1]])
        if st.button("Run"):
            with set_graph_client(client):
                op_param_names = get_op_param_names(fn)
                params = {k: avail_vars[k] for k in op_param_names}
                result = fn(**params)
            result
            st.rerun()
