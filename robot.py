import streamlit as st
import openai
import weave
import random
import string
from typing import Any
from pydantic import Field
from openai.types.chat import (
    ChatCompletionMessageParam,
)
from openai._types import NotGiven
from weave.flow.tools import chat_call_tool_params, perform_tool_calls
from weave.flow.chat_util import OpenAIStream
from weave.trace.refs import parse_uri, OpRef, CallRef
from weave.client_context.weave_client import require_weave_client

from st_components import *

st.set_page_config(layout="wide")

with st.sidebar:
    client = st_project_picker()


def op_from_code(code: str):
    exec_locals = {}
    exec(code, globals(), exec_locals)
    art = MemTraceFilesArtifact({"obj.py": code.encode()})
    fn = op_type.load_instance(art, "obj")

    # Attach art to fn to keep it around in memory, so we can
    # actually read the code file when we go to save!
    fn.art = art
    return fn


@weave.op()
def create_op(code: str) -> str:
    """Create a new op from the given code.

    Args:
        code: Python code for the op we're creating. Must include all necessary imports, and a single python function, decorated with @weave.op().

    Returns:
        A weave ref uri to the created op.
    """
    op = op_from_code(code)
    op_ref = weave.publish(op)
    # Super SUPER gnarly bug. If we return op_ref.uri() here, and the result
    # of this op is stored in a thread, the thread will not deserialize correctly
    # because we automatically convert strings that look like weave refs to
    # *Ref objects. After that you can't pass the thread back to openai.
    # TODO: FIX
    return "Ref: " + op_ref.uri()


def _call_op(op_uri: str, args: list) -> str:
    op_ref = parse_uri(op_uri)
    op = op_ref.get()
    return op(*args)


weave.call_op = _call_op


@weave.op()
def call_op(op_uri: str, args: list) -> str:
    """Call an op with the given uri and arguments.

    Args:
        op_uri: The URI of the op to call.
        args: A list of position arguments to pass to the op.

    Returns:
        A json object containing the ref uri of the resulting call record, and the first 100 characters of the call's output.
    """
    op_ref = parse_uri(op_uri)
    op = op_ref.get()
    result_call = op.call(*args)
    entity, project = result_call.project_id.split("/")
    call_ref = CallRef(entity, project, result_call.id)
    call_output = str(result_call.output)
    if len(call_output) > 100:
        call_output = call_output[:100] + " [truncated]"
    return json.dumps({"call_uri": call_ref.uri(), "call_output_head": call_output})


@weave.op()
def read_call_output_window(call_uri: str, offset: int) -> str:
    """Given a call_uri, read what the output of the call was. This is useful for long outputs. Will return max 1000 chars.

    Args:
        call_uri: The URI of the call to read.
        offset: The offset into the output to start reading from.

    Returns:
        Stringified output starting from offset.
    """
    call_ref = parse_uri(call_uri)
    client = require_weave_client()
    call = client.call(call_ref.id)
    output_s = str(call.output)
    return output_s[offset : offset + 1000]


@weave.op()
def list_ops() -> list[str]:
    """List all ops in the current project.

    Returns:
        A list of URIs for all ops in the current project.
    """
    client = require_weave_client()
    ops = query.get_ops(client)
    return [
        OpRef(client.entity, client.project, op.name, op.digest).uri() for op in ops
    ]


def read_op_code(op_uri: str) -> str:
    """Read the code of an op.

    Args:
        op_uri: The URI of the op to read.

    Returns:
        The code of the op.
    """
    op_ref = parse_uri(op_uri)
    op = op_ref.get()
    print(op.art.path_contents)
    return op.art.path_contents["obj.py"].decode()


class Thread(weave.Object):
    messages: list


class Agent(weave.Object):
    system_message: str
    model_name: str
    temperature: float = 0
    tools: list[Any] = Field(default_factory=list)

    @weave.op()
    def step(self, thread: Thread) -> Thread:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.system_message},
        ]
        messages += thread.messages
        tools = NotGiven()
        if self.tools:
            tools = chat_call_tool_params(self.tools)

        with st.chat_message("assistant"):
            stream = openai.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=messages,
                tools=tools,
                stream=True,
            )
            wrapped_stream = OpenAIStream(stream)
            # OK so this has a side-effect of writing to streamlit?
            # And we're going to track all of our Weave assistant
            st.write_stream(wrapped_stream)
            response = wrapped_stream.final_response()
            response_message = response.choices[0].message
            new_messages = []
            new_messages.append(response_message.model_dump(exclude_none=True))
            if response_message.tool_calls:
                tool_result_messages = perform_tool_calls(
                    self.tools, response_message.tool_calls
                )
                for tool_result_message in tool_result_messages:
                    if "content" in tool_result_message:
                        if tool_result_message["content"].startswith("Ref: "):
                            parsed = parse_uri(
                                tool_result_message["content"].split(" ", 1)[1]
                            )
                            st.session_state.sel_op_ref = parsed
                new_messages += tool_result_messages
        # Still have to do this instead of concat :(
        return Thread(
            name=thread.name,
            messages=thread.messages + new_messages,
        )


thread_objs = query.get_objs(client, types=["Thread"])
if "thread" not in st.session_state:
    if thread_objs:
        st.session_state.thread = thread_objs[0].get()._val
    else:
        st.session_state.thread = Thread(
            name="Thread-"
            + "".join(random.choice(string.ascii_letters) for _ in range(5)),
            messages=[],
        )


if "agent" not in st.session_state:
    st.session_state.agent = Agent(
        system_message="You are so helpful",
        model_name="gpt-4o",
        tools=[create_op, call_op, read_call_output_window, list_ops, read_op_code],
    )


if "sel_op_ref" not in st.session_state:
    st.session_state.sel_op_ref = None


def thread_panel(thread: Thread):
    container = st.container(height=600)
    with container:
        for message in thread.messages:
            with st.chat_message(message["role"]):
                if "content" in message and message["content"]:
                    st.markdown(message["content"])
                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        st.json(tool_call, expanded=False)

    if prompt := st.chat_input("What is up?"):
        thread.messages.append({"role": "user", "content": prompt})

        with container:
            with st.chat_message("user"):
                st.markdown(prompt)

        while True:
            with container:
                thread = st.session_state.agent.step(thread)
            last_message = thread.messages[-1]
            if last_message["role"] == "assistant" and "tool_calls" not in last_message:
                break
        st.session_state.thread = thread


col0, col1 = st.columns(2)
with col0:
    thread_form_cols = st.columns((4, 1))
    with thread_form_cols[0]:

        def on_thread_change():
            st.session_state.thread = st.session_state.thread_picker.get()._val

        st.selectbox(
            "Thread",
            thread_objs,
            format_func=lambda x: f"{x.name} ({x.version_index + 1} steps)",
            key="thread_picker",
            on_change=on_thread_change,
        )
    with thread_form_cols[1]:
        st.markdown(
            "<div style='width: 1px; height: 28px'></div>", unsafe_allow_html=True
        )
        if st.button("New thread"):
            st.session_state.thread = Thread(
                name="Thread-"
                + "".join(random.choice(string.ascii_letters) for _ in range(5)),
                messages=[],
            )
            st.rerun()

    thread_panel(st.session_state.thread)
with col1:
    ops = query.get_ops(client)
    if not ops:
        st.warning("No ops yet")
        st.stop()

    def on_opname_picker_change():
        st.session_state.sel_op_ref = OpRef(
            client.entity,
            client.project,
            st.session_state.opname_picker.name,
            st.session_state.opname_picker.digest,
        )

    for i, op_version in enumerate(ops):
        if (
            st.session_state.sel_op_ref
            and op_version.name == st.session_state.sel_op_ref.name
        ):
            sel_op_name_index = i
            break
    else:
        sel_op_name_index = 0
        st.session_state.sel_op_ref = OpRef(
            client.entity, client.project, ops[0].name, ops[0].digest
        )

    form_cols = st.columns(2)

    with form_cols[0]:
        st.selectbox(
            "Op",
            ops,
            format_func=lambda x: f"{x.name} ({x.version_index + 1} versions)",
            key="opname_picker",
            on_change=on_opname_picker_change,
            index=sel_op_name_index,
        )

    op_version_objs = query.get_op_versions(
        client, st.session_state.sel_op_ref.name, include_call_counts=True
    )

    def on_version_picker_change():
        st.session_state.sel_op_ref = OpRef(
            client.entity,
            client.project,
            st.session_state.op_version_picker.name,
            st.session_state.op_version_picker.digest,
        )

    for i, op_version in enumerate(op_version_objs):
        if (
            st.session_state.sel_op_ref
            and op_version.name == st.session_state.sel_op_ref.name
            and op_version.digest == st.session_state.sel_op_ref.digest
        ):
            sel_op_version_index = i
            break
    else:
        sel_op_version_index = 0
        st.session_state.sel_op_ref = OpRef(
            client.entity,
            client.project,
            op_version_objs[0].name,
            op_version_objs[0].digest,
        )

    with form_cols[1]:
        st.selectbox(
            "Version",
            op_version_objs,
            format_func=lambda x: f"v{x.version_index} ({x.call_count} calls)",
            key="op_version_picker",
            on_change=on_version_picker_change,
            index=sel_op_version_index,
        )

    try:
        op_version_editor(
            client, st.session_state.sel_op_ref.name, st.session_state.sel_op_ref.digest
        )
    except Exception as e:
        st.warning("could not load op version editor: " + str(e))
    calls = query.get_calls(client, [st.session_state.sel_op_ref.uri()])
    if len(calls.df):
        calls_view_df = calls.df[
            [col for col in calls.df.columns if col.startswith("started_at")]
            + [col for col in calls.df.columns if col.startswith("inputs")]
            + [col for col in calls.df.columns if col.startswith("output")]
        ].sort_values("started_at", ascending=False)

        st.write("**Call log**")
        st.dataframe(
            st_safe_df(calls_view_df),
            column_config={
                "started_at": st.column_config.DatetimeColumn(
                    "Started", format="YYYY-MM-DD HH:mm:ss"
                )
            },
        )
