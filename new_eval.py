import streamlit as st
from code_editor import code_editor
import json
import pandas as pd
import openai

import new_api

response_dict = code_editor(
    """

def predict(model_name, math, temp, trial):
    import openai
    result = openai.chat.completions.create(
        model=model_name,
        messages=[{'role': 'user', 'content': math}])
    return result.choices[0].message.content
"""
)

exec_locals = {}
exec(response_dict["text"], {}, exec_locals)
fn = exec_locals[list(exec_locals.keys())[-1]]

sweep = new_api.weave_sweep(fn)
param_names = sweep.param_names()

params = {
    "model_name": json.dumps(["gpt-3.5-turbo", "gpt-4o"]),
    "math": json.dumps(["2 + 2", "2**32"]),
    "temp": json.dumps([0, 1]),
    "trial": json.dumps([0, 1]),
}

for param_name in param_names:
    param_val = st.text_input(param_name, value=params.get(param_name, ""))
    try:
        param_val = json.loads(param_val)
    except json.JSONDecodeError:
        pass
    if param_val is not None:
        params[param_name] = param_val

sweep = sweep(**params)
st.write("stats:", sweep.stats())

submitted = st.button("Run")
if submitted:
    # if submitted:
    # sweep = sweep(**params)
    progress_text = "Computing"
    progress = st.progress(0, text="Computing")

    def on_item_done(frac_done, _, _2):
        if frac_done == 1:
            progress.empty()
        else:
            progress.progress(frac_done, text=progress_text)

    result = sweep.compute(on_item_done=on_item_done)
else:
    result = []
df = pd.json_normalize([{**r[0], "predict": r[1]} for r in result])
df


# def score(math, predict):
#     prompt = f"Does this seem right to you? Respond with just NO or YES.\nQuestion:{math}\nAnswer:{predict}"
#     result = openai.chat.completions.create(
#         model="gpt-4o", messages=[{"role": "user", "content": prompt}]
#     )
#     return result.choices[0].message.content


# with st.spinner("Scoring"):
#     scores = new_api.weave_map(df, score)
# scores_df = pd.json_normalize([{**r[0], "score": r[1]} for r in scores])

# pred_scores = df.merge(scores_df, on=("math", "predict"))


# pivoted = pred_scores.pivot(index=("math", "trial"), columns=("model_name", "temp"))
# pivoted.columns = pivoted.columns.reorder_levels((1, 2, 0))
# pivoted = pivoted.sort_index(axis=1)
# pivoted


# TODO:
#   add Weave caching and make nice state streaming out to streamlit
#   like for example, can we show the pivoted table as it is being computed?
