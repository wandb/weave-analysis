import streamlit as st
import weave
from weave.graph_client_context import set_graph_client
import asyncio

import query
from st_components import (
    nice_ref,
    st_safe_val,
    st_op_selectbox,
    st_wv_column_selectbox,
    st_wv_column_multiselect,
    st_scatter_plot_mean,
    st_barplot_plot_mean,
    st_scatter_pivotxy_mean_histo,
    st_n_histos,
    st_dict,
    st_compare_dict,
    st_multi_dict,
)


@st.cache_resource
def init_weave(project_name: str):
    return weave.init(project_name)


@st.cache_resource
def get_ref(_client, ref):
    with st.spinner("ref.get"):
        return _client.get(ref)


with st.spinner("weave.init"):
    wv = init_weave("weavedev-hooman2")

models = query.get_objs(wv, types="Model")

model_ref = st.selectbox(
    "Model", models, format_func=lambda model: f"{model.name}:{model.digest}"
)

model = get_ref(wv, model_ref)
model_val = model._val.__dict__
# model_val

with st.form("call-form"):
    input_val = st.text_input("Input", value="hello")
    submitted = st.form_submit_button("Predict")

if submitted:
    with st.spinner("calling model predict"):
        with set_graph_client(wv):
            result = asyncio.run(model.predict(input_val))
    result

# with st.spinner("calling model predict"):
#     result = asyncio.run(model.predict("hello"))
# result
# simple_model = query.simple_val(model._val)
# simple_model
