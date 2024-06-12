import streamlit as st
import pandas as pd
import plotly.express as px
import weave
import json
import math

from weave_api_next import weave_client_ops, weave_client_calls, weave_client_get_batch

st.set_page_config(layout="wide")


DEFAULT_PROJECT_NAME = "humaneval6"


@st.cache_data()
def get_ops(project_name):
    client = weave.init(project_name)
    return weave_client_ops(client)


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
    return pd.json_normalize(call_list)


@st.cache_data()
def resolve_refs(example_refs):
    client = weave.init(DEFAULT_PROJECT_NAME)
    # Resolve the refs and fetch the message.text field
    # Note we do do this after grouping, so we don't over-fetch refs
    return weave_client_get_batch(client, example_refs)


def is_ref_column(df, column):
    return df[column].str.startswith("weave://").any()


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


def expanded_ref_column(df, column):
    ref_vals = resolve_refs(df[column])
    ref_vals = simple_val(ref_vals)
    ref_val_df = pd.json_normalize(ref_vals)
    ref_val_df.index = df.index
    return ref_val_df


def expand_ref_column(df, column):
    ref_val_df = expanded_ref_column(df, column)
    # add ref vals to df with keys <column>.<key>
    column_index = df.columns.get_loc(column)
    df = df.copy()
    for i, key in enumerate(ref_val_df.columns):
        df.insert(column_index + i + 1, f"{column}.{key}", ref_val_df[key])
    return df


def parse_ref(x):
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

        return f"{parsed.name}:{parsed.digest[:3]}"
    except ValueError:
        return x


def st_safe_df(df):
    # Streamlit dies on some pyarrow code if there is a list column that
    # has non-uniform types in it. So just convert those to json strings
    # for display

    def to_json_string(val):
        if isinstance(val, list):
            return json.dumps(val)
        return val

    # Apply the function to each element of the DataFrame
    df = df.applymap(to_json_string)
    return df


def st_write_dict(d):
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


def st_write_compare_dict(dicts, headers):
    keys = set()
    for d in dicts:
        keys.update(d.keys())

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
        col.write(f"**{value}**")

    show_diff = st.toggle("Differences", value=True)
    if show_diff:
        for key in diff_keys:
            cols = st.columns([1] + [2] * len(dicts))
            cols_iter = iter(cols)
            col0 = next(cols_iter)
            col0.write(f"**{key}**")
            for col, value in zip(cols_iter, [d.get(key, None) for d in dicts]):
                col.write(value)

    show_common = st.toggle("Common Values")
    if show_common:
        for key in common_keys:
            cols = st.columns((1, 2 * len(dicts)))
            cols[0].write(f"**{key}**")
            cols[1].write(dicts[0].get(key))


with st.sidebar:
    # project_name = st.selectbox("Project Name", ["humaneval6", "weave-hooman1"])
    project_name = st.text_input("Project Name", "weave-hooman1")

    objs = get_ops(project_name)

    op_objs = sorted(
        objs,
        key=lambda o: (
            not "predict_and_score" in o.object_id,
            o.object_id,
        ),
    )

    op_obj = st.selectbox(
        "Op",
        op_objs,
        format_func=lambda x: f"{x.object_id} ({x.version_index + 1} versions)",
    )

    # op_version = st.selectbox(
    #     "Version",
    #     ["*"] + [f"v{v}" for v in range(op_obj.version_index + 1)],
    # )

    calls = get_calls(project_name, op_obj.object_id)
    calls_dtypes = friendly_dtypes(calls)

    # Input filters
    # input_keys = [c.split(".", 1)[1] for c in calls.columns if c.startswith("inputs.")]
    # selection = {}
    # with st.expander("input filters"):
    #     for input_key in input_keys:
    #         selection[input_key] = st.multiselect(
    #             input_key, calls[f"inputs.{input_key}"].unique()
    #         )
    # Filter calls to selection
    # for key, values in selection.items():
    #     if values:
    #         calls = calls[calls[f"inputs.{key}"].isin(values)]

    ordered_compare_columns = sorted(
        (
            c
            for c in calls_dtypes.index
            if pd.api.types.is_string_dtype(calls_dtypes[c])
        ),
        key=lambda col: (
            not col == "inputs.model",
            not col.startswith("inputs."),
            col,
        ),
    )

    compare_key = st.selectbox("A / B compare", ordered_compare_columns)
    compare_key_render = f"{compare_key}.render"
    if is_ref_column(calls, compare_key):
        calls[compare_key_render] = calls[compare_key].apply(parse_ref)
    else:
        calls[compare_key_render] = calls[compare_key]

    ordered_target_columns = sorted(
        (
            c
            for c in calls_dtypes.index
            if (
                pd.api.types.is_numeric_dtype(calls_dtypes[c])
                or pd.api.types.is_bool_dtype(calls_dtypes[c])
            )
        ),
        key=lambda col: (
            # not pd.api.types.is_string_dtype(calls_dtypes[col]),
            not col.startswith("outputs."),
            col,
        ),
    )
    target_keys = st.multiselect("Target", ordered_target_columns)
    ordered_compare_columns = sorted(
        (
            c
            for c in calls_dtypes.index
            if pd.api.types.is_string_dtype(calls_dtypes[c])
        ),
        key=lambda col: (
            not col == "inputs.example",
            not col.startswith("inputs."),
            col,
        ),
    )
    across_key = st.selectbox("Across", ordered_compare_columns)

##### Per compare val plot, showing target0 v. target1 #####

compare_vals = calls[compare_key].unique()
compare_val_stats_df = calls.groupby(compare_key).agg(
    {compare_key_render: "first", **{k: ["mean", "sem"] for k in target_keys}}
)
compare_val_stats_df.columns = [".".join(col) for col in compare_val_stats_df.columns]
compare_val_stats_df.rename(
    {f"{compare_key_render}.first": compare_key_render}, axis=1, inplace=True
)
compare_val_stats_df[compare_key] = compare_val_stats_df.index

st.header(f"Comparing *{op_obj.object_id}* calls by *{compare_key}*")


if len(target_keys) < 2:
    st.warning("Select at least two target keys")
    st.stop()
target0 = target_keys[0]
target1 = target_keys[1]


fig = px.scatter(
    compare_val_stats_df,
    x=f"{target1}.mean",
    error_x=f"{target1}.sem",
    y=f"{target0}.mean",
    error_y=f"{target0}.sem",
    custom_data=[compare_key],
    labels={compare_key_render: compare_key},
    color=compare_key_render,
)
fig.update_layout(dragmode="select")
selected = st.plotly_chart(fig, on_select="rerun")

##### Show the selected points, only 2 for now #####

compare_vals = [p["customdata"][0] for p in selected["selection"]["points"]]

if len(compare_vals) < 2:
    st.warning("Select at least two points to compare on the chart above")
    st.stop()

compare_vals_render = [
    # streamlit write/text removes stuff after ":", so escape it
    str(compare_val_stats_df.loc[compare_val, compare_key_render]).replace(":", "\\:")
    for compare_val in compare_vals
]
compare_val0 = compare_vals[0]
compare_val0_render = compare_vals_render[0]
compare_val1 = compare_vals[1]
compare_val1_render = compare_vals_render[1]

if is_ref_column(compare_val_stats_df, compare_key):
    # compare_val_stats_df = expand_ref_column(compare_val_stats_df, compare_key)
    expanded_df = expanded_ref_column(compare_val_stats_df, compare_key)
    compare_vals_df = expanded_df.loc[compare_vals,]
    with st.expander(
        f"Comparing **{compare_val0_render}** and **{compare_val1_render}** [First 2 of {len(compare_vals_df)} selected]"
    ):
        st_write_compare_dict(
            compare_vals_df.to_dict(orient="records")[:2], compare_vals_render[:2]
        )

##### Second level comparison by across_key #####

st.header(f"Split by *{across_key}*")

model_calls = calls[calls[compare_key].isin([compare_val0, compare_val1])]

calls = model_calls

# Build a dataframe with one row per across value

example_uniqs = calls[across_key].unique()
example_df = pd.Series(example_uniqs, index=example_uniqs).to_frame(name=across_key)
if is_ref_column(example_df, across_key):
    example_df = expand_ref_column(example_df, across_key)


# Add columns with average value for each target key
per_model_pivot_df = calls.pivot_table(
    index=across_key,
    columns=compare_key,
    values=target_keys,
    aggfunc=("mean",),
)
per_model_pivot_df = per_model_pivot_df.swaplevel(1, 2, axis=1)
example_df.columns = pd.MultiIndex.from_frame(
    example_df.columns.to_frame().assign(second="", third="")
)
example_df = pd.concat((example_df, per_model_pivot_df), axis=1)

# Add columns, one for each compare output offset
output_column_names = calls.filter(like="output.").columns
model_preds = calls[[across_key, compare_key] + output_column_names.tolist()]
model_preds["model_output_index"] = model_preds.groupby(
    [across_key, compare_key]
).cumcount()
pivot_df = model_preds.pivot(
    index=across_key,
    columns=[compare_key, "model_output_index"],
    values=output_column_names,
)
example_df = example_df.join(pivot_df)


n_plot_cols = 2
plot_cols = st.columns(n_plot_cols)
plot_selected_indexes = []
plot_df = example_df.copy()
plot_df["index"] = plot_df.index
plot_df.columns = [".".join((str(c) for c in col if c)) for col in plot_df.columns]
for i, t in enumerate(target_keys):
    plot_col = plot_cols[i % n_plot_cols]
    x_col = f"{t}.{compare_val0}.mean"
    y_col = f"{t}.{compare_val1}.mean"
    aggregated_df = plot_df.groupby([x_col, y_col]).size().reset_index(name="count")
    aggregated_df["index"] = aggregated_df.index

    fig = px.scatter(
        aggregated_df,
        x=x_col,
        y=y_col,
        labels={
            x_col: compare_val0_render,
            y_col: compare_val1_render,
        },
        size="count",
        custom_data="index",
        title=t,
    )
    fig.update_layout(dragmode="select")
    selected = plot_col.plotly_chart(fig, on_select="rerun")
    selected_indexes = [p["customdata"][0] for p in selected["selection"]["points"]]
    if selected_indexes:
        selected = aggregated_df.iloc[selected_indexes]
        plot_df = plot_df.merge(selected[[x_col, y_col]], on=[x_col, y_col])

safe_plot_df = st_safe_df(plot_df)
row_selection = st.dataframe(
    safe_plot_df, on_select="rerun", selection_mode="single-row"
)

selected_rows = row_selection["selection"]["rows"]
if not selected_rows:
    st.warning("Select a row to view details.")
    st.stop()

plot_selected_row = plot_df.iloc[selected_rows[0]]
selected_row = example_df.loc[plot_selected_row["index"]]


# Convert the filtered Series to a dictionary
inputs = selected_row[selected_row.index.get_level_values(0).str.startswith("inputs.")]
inputs = {k[0]: v for k, v in inputs.to_dict().items()}
st.subheader("input")
with st.expander("input value"):
    show_keys = st.multiselect("keys", inputs.keys())
    if show_keys:
        inputs = {k: inputs[k] for k in show_keys}
    st_write_dict(inputs)


st.subheader("output")

col1, col2 = st.columns(2)
with col1:
    st.write(compare_val0_render)
with col2:
    st.write(compare_val1_render)

for index in selected_row.index.get_level_values(2).unique():
    if not isinstance(index, int):
        continue
    with st.container():
        st.subheader(index, divider=True)
        col1, col2 = st.columns(2)
        with col1:
            model0_outputs = selected_row.loc[
                (slice(None), compare_val0, index)
            ].dropna()
            st_write_dict(model0_outputs)
        with col2:
            model1_outputs = selected_row.loc[
                (slice(None), compare_val1, index)
            ].dropna()
            st_write_dict(model1_outputs)


##### Other plots below #####

# call record counts over time plot
# # Create a new column for exception status
# calls["exception_status"] = calls["exception"].apply(
#     lambda x: "exception" if pd.notnull(x) else "No Exception"
# )
# # Create the histogram
# fig = px.histogram(
#     calls,
#     x="started_at",
#     color="exception_status",
#     barmode="group",
#     labels={
#         "started_at": "Date",
#         "count": "Call Count",
#         "exception_status": "Exception Status",
#     },
#     title="Call Record Counts Over Time by Exception Status",
# )
# Show the plot
# st.plotly_chart(fig)

# correct bar plot
# fig = px.histogram(
#     calls,
#     y="inputs.model",
#     color="output.scores.score_humaneval_test.correct",
#     barmode="group",
# )
# st.plotly_chart(fig)
