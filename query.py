from typing import Callable, Optional, Any
from dataclasses import dataclass
import pandas as pd
import weave
import streamlit as st
import math

from weave_api_next import weave_client_ops, weave_client_calls, weave_client_get_batch


def simple_val(v):
    if isinstance(v, dict):
        return {k: simple_val(v) for k, v in v.items()}
    elif isinstance(v, list):
        return [simple_val(v) for v in v]
    elif hasattr(v, "uri"):
        return v.uri()
    elif hasattr(v, "__dict__"):
        return {k: simple_val(v) for k, v in v.__dict__.items()}
    else:
        return v


def is_ref_series(series: pd.Series):
    return series.str.startswith("weave://").any()


@st.cache_data()
def resolve_refs(project_name, refs):
    client = weave.init(project_name)
    # Resolve the refs and fetch the message.text field
    # Note we do do this after grouping, so we don't over-fetch refs
    ref_vals = weave_client_get_batch(client, refs)
    ref_vals = simple_val(ref_vals)
    ref_val_df = pd.json_normalize(ref_vals)
    ref_val_df.index = refs
    return ref_val_df


@dataclass
class Op:
    name: str
    version_index: int


@st.cache_data()
def get_ops(project_name):
    client = weave.init(project_name)
    client_ops = weave_client_ops(client)
    return [Op(op.object_id, op.version_index) for op in client_ops]


def friendly_dtypes(df):
    # Pandas doesn't allow NaN in bool columns for example. But we want to suggest
    # bool-like columns as target columns for example.
    def detect_dtype(series):
        non_null_series = series.dropna()

        if non_null_series.empty:
            return "unknown"

        # Check for boolean-like columns
        if all(
            isinstance(x, bool) or x is None or (isinstance(x, float) and math.isnan(x))
            for x in series
        ):
            return "bool"

        # Check for string-like columns
        if all(
            isinstance(x, str) or x is None or (isinstance(x, float) and math.isnan(x))
            for x in series
        ):
            return "str"

        # Fallback to the series' original dtype
        return series.dtype.name

    dtypes_dict = {col: detect_dtype(df[col]) for col in df.columns}
    friendly_dtypes_series = pd.Series(dtypes_dict, name="Friendly Dtype")
    return friendly_dtypes_series


@dataclass
class Column:
    name: str
    type: str


@dataclass
class Calls:
    df: pd.DataFrame

    def columns(
        # TODO what is the python type for sorted key return value?
        self,
        op_types=None,
        sort_key: Optional[Callable[[Column], Any]] = None,
    ):
        dtypes = friendly_dtypes(self.df)
        cols = (Column(c, dtypes[c]) for c in dtypes.index)
        if op_types:
            cols = (c for c in cols if dtypes[c.name] in op_types)
        if sort_key:
            cols = sorted(cols, key=sort_key)
        return cols


@st.cache_data()
def get_calls(project_name, op_name):
    client = weave.init(project_name)
    call_list = [
        {
            "id": c.id,
            "trace_id": c.trace_id,
            "parent_id": c.parent_id,
            "inputs": {
                k: v.uri() if hasattr(v, "uri") else v for k, v in c.inputs.items()
            },
            "output": c.output,
            "exception": c.exception,
            # "attributes": c.attributes,
            "summary": c.summary,
            # "started_at": c.started_at,
            # "ended_at": c.ended_at,
        }
        for c in weave_client_calls(client, op_name)
    ]
    return Calls(pd.json_normalize(call_list))
