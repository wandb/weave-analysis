from functools import wraps
import openai
from st_components import *

from api2.query_api import calls
from api2.engine import init_engine
from api2.query_api import Query
from api2.execute_api import weave_map


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

my_calls = calls(wc, op, limit=10)
example = my_calls.column("inputs.example").expand_ref()


@weave.op()
def classify(code: str):
    prompt = f"Classify the following code. Return a single word.\n\n{code}\n\nClass:"
    response = openai.chat.completions.create(
        model="gpt-4o", messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content


mapper = weave_map(example, classify, {"code": "prompt"})
st.dataframe(example, hide_index=True)

cost = mapper.cost()
st.write(cost)
if cost["to_compute"]:
    if st.button("Execute"):
        prog = st.progress(0, "computing")
        for i, delta in enumerate(mapper.execute()):
            prog.progress((i + 1) / cost["to_compute"])
        st.rerun()
else:
    st.dataframe(mapper.get_result())

# TODO: OK this works. Now I need to get all of it working in one table
# so I can do a groupby

# sel_models = wv_st_scatter_plot_mean(
#     my_calls,
#     "inputs.model",
#     t1,
#     t0,
# )

# sel_examples = wv_st_scatter_plot_mean(sel_models.groups(), "inputs.example", t1, t0)

# st.dataframe(sel_examples.groups(), hide_index=True)

# df_sel = wv_st_dataframe(sel)
# # df_sel.group_vals()

# # OK I'm trying to figure out how to pass group values through. We want to
# # display the original call list here.
# if df_sel:
#     x = df_sel.groups().to_pandas()
#     x
# st.dataframe(df_sel)
# st.json(df_sel.to_pandas().to_dict())
