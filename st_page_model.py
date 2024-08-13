import streamlit as st
import weave
import asyncio

from weave.trace.refs import parse_uri

import query
from st_components import *


def get_ref(client, ref):
    with st.spinner("ref.get"):
        return client.get(ref)


wv = st_project_picker()

models = query.get_objs(wv, types="Model")
model_refs = [parse_uri(r) for r in models.index]

model_ref = st.selectbox(
    "Model",
    model_refs,
    format_func=lambda model: f"{model.name}:{model.digest[:3]}",
)

model = get_ref(wv, model_ref)
model_val = model._val.__dict__
# model_val

with st.form("call-form"):
    input_val = st.text_input("Input", value="hello")
    submitted = st.form_submit_button("Predict")

if submitted:
    with st.spinner("calling model predict"):
        result = asyncio.run(model.predict(input_val))
    result

# with st.spinner("calling model predict"):
#     result = asyncio.run(model.predict("hello"))
# result
# simple_model = query.simple_val(model._val)
# simple_model
