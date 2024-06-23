import streamlit as st

import weave
import new_api
import openai


st.set_page_config(layout="wide")

weave.init_local_client()


@st.cache_resource
def init_weave():
    return weave.init_local_client()


client = init_weave()


@weave.op()
def answer(model_name, question, temp, trial):
    result = openai.chat.completions.create(
        temperature=temp,
        model=model_name,
        messages=[{"role": "user", "content": question}],
        # stream=True,
    )
    return result.choices[0].message.content
    return result


@weave.op()
def guess_question(answer):
    prompt = f"What is this an answer to, just guess one thing: {answer}"
    result = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],  # stream=True
    )
    return result.choices[0].message.content
    return result


@weave.op()
def score_guess(question, guess_question):
    prompt = f"Does the question in this guess '{guess_question}' match the original question '{question}'? Just say YES or NO"
    result = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],  # stream=True
    )
    return result.choices[0].message.content
    return result


run = st.button("Run")

if run:
    t = new_api.ComputeTable()

    t.add_input("model_name")
    t.add_input("question")
    t.add_input("temp")
    t.add_input("trial")

    t.add_op(answer)
    t.add_op(guess_question)
    t.add_op(score_guess)

    t.add_row(
        {"model_name": "gpt-3.5-turbo", "question": "2 + 2", "temp": 0, "trial": 0}
    )
    t.add_row(
        {"model_name": "gpt-3.5-turbo", "question": "2 ** 32", "temp": 0, "trial": 0}
    )
    t.add_row(
        {"model_name": "gpt-3.5-turbo", "question": "x *** y", "temp": 0, "trial": 0}
    )
    t.add_row(
        {
            "model_name": "gpt-3.5-turbo",
            "question": "how are you feeling today?",
            "temp": 0,
            "trial": 0,
        }
    )
    t.add_row(
        {
            "model_name": "gpt-3.5-turbo",
            "question": "could time be circular?",
            "temp": 0,
            "trial": 0,
        }
    )

    containers = {}

    col_widths = [
        1 if isinstance(col, new_api.InputColumn) else 3 for col in t.columns.values()
    ]

    for st_col, (col_name, col) in zip(st.columns(col_widths), t.columns.items()):
        with st_col:
            st.write(f"**{col_name}**")
    st.divider()
    for i, row in enumerate(t.rows):
        for st_col, (col_name, col) in zip(st.columns(col_widths), t.columns.items()):
            with st_col:
                with st.container(height=100, border=False):
                    empty = st.empty()
                    containers[(i, col_name)] = empty

                    empty.write(row[col_name])
        st.divider()

    for delta in t.execute():
        cont = containers[delta["row"], delta["col"]]
        cont.empty()
        cont.write(delta["val"])
