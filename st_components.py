from typing import Callable, Optional, Any, Sequence
import pandas as pd
import plotly.express as px
import streamlit as st
import json
import query
import weave
import numpy as np
from code_editor import code_editor
from weave.trace_server.trace_server_interface import (
    FileContentReadReq,
    ObjReadReq,
)
from weave.trace.custom_objs import MemTraceFilesArtifact
from weave.trace import op_type
from weave.trace.refs import OpRef


def nice_ref(x):
    try:
        parsed = weave.trace.refs.parse_uri(x)
        nice_name = f"{parsed.name}:{parsed.digest[:3]}"
        if parsed.extra:
            for i in range(0, len(parsed.extra), 2):
                k = parsed.extra[i]
                v = parsed.extra[i + 1]
                if k == "id":
                    nice_name += f"/{v[:4]}"
        return nice_name
    except ValueError:
        return x


def st_safe_val(val):
    if isinstance(val, str):
        return val.replace(":", "\\:")
    return val


def st_safe_df(df):
    # Streamlit dies on some pyarrow code if there is a list column that
    # has non-uniform types in it. So just convert those to json strings
    # for display

    def to_json_string(val):
        if isinstance(val, list):
            try:
                return json.dumps(val)
            except TypeError:
                return str(val)
        return val

    # Apply the function to each element of the DataFrame
    df = df.applymap(to_json_string)
    return df


def st_op_selectbox(
    client,
    label: str,
    sort_key: Optional[Callable[[query.Op], Any]] = None,
):
    import streamlit as st

    ops = query.get_ops(client)
    if sort_key:
        ops = sorted(ops, key=sort_key)
    return st.selectbox(
        label, ops, format_func=lambda x: f"{x.name} ({x.version_index + 1} versions)"
    )


def st_wv_column_selectbox(
    label: str,
    calls: query.Calls,
    op_types: Optional[Sequence[str]] = None,
    sort_key: Optional[Callable[[query.Column], Any]] = None,
):
    ordered_compare_cols = calls.columns(op_types=op_types, sort_key=sort_key)
    ordered_compare_column_names = [col.name for col in ordered_compare_cols]
    return st.selectbox(label, ordered_compare_column_names)


def st_wv_column_multiselect(
    label: str,
    calls: query.Calls,
    op_types: Optional[Sequence[str]] = None,
    sort_key: Optional[Callable[[query.Column], Any]] = None,
    default: Optional[Callable[[Sequence[str]], Any]] = None,
):
    ordered_compare_cols = calls.columns(op_types=op_types, sort_key=sort_key)
    ordered_compare_column_names = [col.name for col in ordered_compare_cols]
    if default is not None:
        default_val = default(ordered_compare_column_names)
    return st.multiselect(label, ordered_compare_column_names, default=default_val)


def wv_st_scatter_plot_mean(df: pd.DataFrame, compare_key: str, x_key: str, y_key: str):
    compare_val_stats_df = df.groupby(compare_key).agg(
        {x_key: ["mean", "sem"], y_key: ["mean", "sem"]}
    )
    compare_val_stats_df.columns = [
        ".".join(col) for col in compare_val_stats_df.columns
    ]
    compare_key_render = compare_key + ".render"
    compare_val_stats_df[compare_key_render] = compare_val_stats_df.index.map(nice_ref)
    compare_val_stats_df[compare_key] = compare_val_stats_df.index

    fig = px.scatter(
        compare_val_stats_df,
        x=f"{x_key}.mean",
        error_x=f"{x_key}.sem",
        y=f"{y_key}.mean",
        error_y=f"{y_key}.sem",
        custom_data=[compare_key],
        labels={compare_key_render: compare_key},
        color=compare_key_render,
    )
    fig.update_layout(dragmode="select")

    selected = st.plotly_chart(fig, on_select="rerun")
    selected_vals = [p["customdata"][0] for p in selected["selection"]["points"]]

    return compare_val_stats_df, selected_vals


def st_barplot_plot_mean(df: pd.DataFrame, compare_key: str, x_key: str):
    compare_val_stats_df = df.groupby(compare_key).agg({x_key: ["mean", "sem"]})
    compare_val_stats_df.columns = [
        ".".join(col) for col in compare_val_stats_df.columns
    ]
    compare_key_render = compare_key + ".render"
    compare_val_stats_df[compare_key_render] = compare_val_stats_df.index.map(nice_ref)
    compare_val_stats_df[compare_key] = compare_val_stats_df.index

    fig = px.bar(
        compare_val_stats_df,
        x=f"{x_key}.mean",
        y=compare_key_render,
        orientation="h",
        custom_data=[compare_key],
    )
    fig.update_layout(dragmode="select")

    selected = st.plotly_chart(fig, on_select="rerun")
    selected_vals = [p["customdata"][0] for p in selected["selection"]["points"]]

    return compare_val_stats_df, selected_vals


def st_xy_histo(df: pd.DataFrame, x_key: str, y_key: str, title: str):
    histo_df = df.groupby([x_key, y_key]).size().reset_index(name="count")
    histo_df["index"] = histo_df.index
    fig = px.scatter(
        histo_df,
        x=x_key,
        y=y_key,
        labels={
            x_key: nice_ref(x_key),
            y_key: nice_ref(y_key),
        },
        size="count",
        custom_data="index",
        title=title,
    )
    fig.update_layout(dragmode="select")
    selected = st.plotly_chart(fig, on_select="rerun", selection_mode="box")
    selected_vals = [p["customdata"][0] for p in selected["selection"]["points"]]
    selected_histo = histo_df.loc[selected_vals]
    df_index = df.reset_index(names="index")
    merged_df = df_index.merge(selected_histo[[x_key, y_key]], on=[x_key, y_key])
    return merged_df["index"]


def st_scatter_pivotxy_mean_histo(
    df: pd.DataFrame,
    compare_key: str,
    pivot_key: str,
    value_key: str,
    x_key: str,
    y_key: str,
    title: str,
):
    pivot_df = df.pivot_table(
        index=compare_key, columns=pivot_key, values=value_key, aggfunc="mean"
    )

    selected = st_xy_histo(pivot_df, x_key=x_key, y_key=y_key, title=title)
    return selected


def st_n_histos(df: pd.DataFrame, compare_key: str, n_key: str, x_key: str):
    figs = []
    num_bins = 20

    bin_edges = np.histogram_bin_edges(df[x_key].dropna(), bins=num_bins)
    n_fs = []
    hists = []
    for n_val in df[n_key].unique():
        n_df = df[df[n_key] == n_val]
        n_fs.append(n_df)
        hist = np.histogram(n_df[x_key], bins=bin_edges)
        hists.append(hist)
    max_y = max(hist[0].max() for hist in hists)
    for n_df, hist in zip(n_fs, hists):
        compare_val_stats_df = n_df.groupby(compare_key).agg({x_key: ["mean", "sem"]})
        compare_val_stats_df.columns = [
            ".".join(col) for col in compare_val_stats_df.columns
        ]
        compare_key_render = compare_key + ".render"
        compare_val_stats_df[compare_key_render] = compare_val_stats_df.index.map(
            nice_ref
        )
        compare_val_stats_df[compare_key] = compare_val_stats_df.index

        fig = px.histogram(
            compare_val_stats_df,
            x=f"{x_key}.mean",
            nbins=num_bins,
            range_y=(0, max_y),
        )
        fig.update_layout(dragmode="select")
        figs.append(fig)

    return figs


def st_dict(d, header=None):
    if header:
        header_val = nice_ref(header).replace(":", "\\:")
        st.write(f"**{header_val}**")
    for key, value in d.items():
        if isinstance(value, str):
            st.write(f"**{key}**")
            st.write(value)
        elif isinstance(value, dict):
            st.write(f"**{key}**")
            st.json(value, expanded=False)
        elif isinstance(value, list):
            st.write(f"**{key}**")
            st.json(value, expanded=False)
        elif isinstance(value, float):
            st.write(f"**{key}**", value)
        elif isinstance(value, bool):
            st.write(f"**{key}**", value)


def st_multi_dict(dicts, headers):
    keys = {}
    for d in dicts:
        for k in d.keys():
            keys[k] = True
    keys = list(keys)

    cols = st.columns([1] + [2] * len(dicts))
    cols_iter = iter(cols)
    col0 = next(cols_iter)
    for col, value in zip(cols_iter, headers):
        value = nice_ref(value).replace(":", "\\:")
        col.write(f"**{value}**")

    for key in keys:
        cols = st.columns([1] + [2] * len(dicts))
        cols_iter = iter(cols)
        col0 = next(cols_iter)
        col0.write(f"**{key}**")
        for col, value in zip(cols_iter, [d.get(key, None) for d in dicts]):
            col.write(value)


def st_compare_dict(dicts, headers, st_key="compare"):
    keys = {}
    for d in dicts:
        for k in d.keys():
            keys[k] = True
    keys = list(keys)

    diff_keys = []
    common_keys = []

    for key in keys:
        values = [d.get(key, None) for d in dicts]
        if all(v == values[0] for v in values):
            common_keys.append(key)
        else:
            diff_keys.append(key)

    cols = st.columns([1] + [2] * len(dicts))
    cols_iter = iter(cols)
    col0 = next(cols_iter)
    for col, value in zip(cols_iter, headers):
        value = nice_ref(value).replace(":", "\\:")
        col.write(f"**{value}**")

    show_diff = st.toggle("Differences", value=True, key=st_key + "-diff")
    if show_diff:
        for key in diff_keys:
            cols = st.columns([1] + [2] * len(dicts))
            cols_iter = iter(cols)
            col0 = next(cols_iter)
            col0.write(f"**{key}**")
            for col, value in zip(cols_iter, [d.get(key, None) for d in dicts]):
                col.write(value)

    show_common = st.toggle("Common Values", key=st_key + "-common")
    if show_common:
        for key in common_keys:
            cols = st.columns((1, 2 * len(dicts)))
            cols[0].write(f"**{key}**")
            cols[1].write(dicts[0].get(key))


def op_code_editor(op_code: str, key: str):
    response_dict = code_editor(op_code, key=key)
    if response_dict["text"]:
        op_code = response_dict["text"]
    exec_locals = {}
    exec(op_code, globals(), exec_locals)
    art = MemTraceFilesArtifact({"obj.py": op_code.encode()})
    fn = op_type.load_instance(art, "obj")

    # Attach art to fn to keep it around in memory, so we can
    # actually read the code file when we go to save!
    fn.art = art
    return fn


def op_version_editor(client, op_name, version):
    fn = None
    server_read = client.server.obj_read(
        ObjReadReq(project_id=client._project_id(), object_id=op_name, digest=version)
    )
    code_file_digest = server_read.obj.val["files"]["obj.py"]
    code_file_contents = client.server.file_content_read(
        FileContentReadReq(project_id=client._project_id(), digest=code_file_digest)
    ).content.decode()
    fn = op_code_editor(code_file_contents, f"{op_name}:{version}")

    return fn


@st.cache_resource
def init_local_weave(db_path: str):
    return weave.init_local_client(db_path)


@st.cache_resource
def init_remote_weave(project: str):
    return weave.init(project)


def st_project_picker():
    source = st.radio("project source", ["local", "wandb"])
    if source == "wandb":
        project_name = st.text_input("project name")
        return init_remote_weave(project_name)
    else:
        db_file = st.text_input("db file path", value="example_eval.db")
        return init_local_weave(db_file)
