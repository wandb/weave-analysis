from functools import wraps
from st_components import *

from api2.provider import calls
from api2.engine import init_engine
from api2.provider import Query


def wv_st_plotly_chart(data, fig):
    fig.update_layout(dragmode="select")
    selected = st.plotly_chart(fig, on_select="rerun")
    selected_vals = [p["customdata"][0] for p in selected["selection"]["points"]]
    return data.loc(selected_vals)


def wv_st_dataframe(data):
    data_df = data.to_pandas()
    selected = st.dataframe(data_df, selection_mode="single-row", on_select="rerun")
    selected_rows = selected["selection"]["rows"]
    if selected_rows:
        selected_row = selected_rows[0]
        row = data_df.index[selected_row]
        return data.loc(row)
    return


def wv_st_scatter_plot_mean(input: Query, compare_key: str, x_key: str, y_key: str):
    stats = input.groupby(compare_key).agg(
        {x_key: ["mean", "sem"], y_key: ["mean", "sem"]}
    )
    stats_df = stats.to_pandas()
    compare_key_render = compare_key + ".render"
    stats_df[compare_key_render] = stats_df.index.map(nice_ref)
    fig = px.scatter(
        stats_df,
        x=f"{x_key}.mean",
        error_x=f"{x_key}.sem",
        y=f"{y_key}.mean",
        error_y=f"{y_key}.sem",
        custom_data=[compare_key],
        labels={compare_key_render: compare_key},
        color=compare_key_render,
    )
    return wv_st_plotly_chart(stats, fig)


proj = "humaneval6"
op = "Evaluation.predict_and_score"
targets = [
    "output.scores.score_humaneval_test.correct",
    "output.model_latency",
]
t0 = targets[0]
t1 = targets[1]

wc = init_remote_weave("humaneval6")
init_engine(wc)

my_calls = calls(wc, op, limit=1000)


sel_models = wv_st_scatter_plot_mean(
    my_calls,
    "inputs.model",
    t1,
    t0,
)

sel_examples = wv_st_scatter_plot_mean(sel_models.groups(), "inputs.example", t1, t0)

st.dataframe(sel_examples.groups(), hide_index=True)

# df_sel = wv_st_dataframe(sel)
# # df_sel.group_vals()

# # OK I'm trying to figure out how to pass group values through. We want to
# # display the original call list here.
# if df_sel:
#     x = df_sel.groups().to_pandas()
#     x
# st.dataframe(df_sel)
# st.json(df_sel.to_pandas().to_dict())
