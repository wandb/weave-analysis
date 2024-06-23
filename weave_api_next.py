# This contains code that needs to be added to the weave Python package.
# But its here for now so we can iterate on finding the right patterns.

import dataclasses
from typing import cast, Optional, Union, Iterator, Sequence, Any
from weave.weave_client import WeaveClient, from_json
from weave import urls
from weave.trace_server.trace_server_interface import (
    ObjSchema,
    ObjQueryReq,
    _ObjectVersionFilter,
    _CallsFilter,
    CallsQueryReq,
    TraceServerInterface,
    CallSchema,
    RefsReadBatchReq,
)
from weave.trace_server.trace_server_interface_util import extract_refs_from_values
from weave.trace.vals import TraceObject
from weave import graph_client_context


@dataclasses.dataclass
class Call:
    op_name: str
    trace_id: str
    project_id: str
    parent_id: Optional[str]
    inputs: dict
    input_refs: list[str]
    id: Optional[str] = None
    output: Any = None
    exception: Optional[str] = None
    summary: Optional[dict] = None
    # These are the live children during logging
    _children: list["Call"] = dataclasses.field(default_factory=list)

    @property
    def ui_url(self) -> str:
        project_parts = self.project_id.split("/")
        if len(project_parts) != 2:
            raise ValueError(f"Invalid project_id: {self.project_id}")
        entity, project = project_parts
        if not self.id:
            raise ValueError("Can't get URL for call without ID")
        return urls.redirect_call(entity, project, self.id)

    # These are the children if we're using Call at read-time
    def children(self) -> "CallsIter":
        client = graph_client_context.require_graph_client()
        if not self.id:
            raise ValueError("Can't get children of call without ID")
        return CallsIter(
            client.server,
            self.project_id,
            _CallsFilter(parent_ids=[self.id]),
        )

    def delete(self) -> bool:
        client = graph_client_context.require_graph_client()
        return client.delete_call(call=self)


def make_client_call(
    entity: str, project: str, server_call: CallSchema, server: TraceServerInterface
) -> Call:
    output = server_call.output
    # extract_refs_from_values operates on strings. We could return ref objects
    # here instead, since those are what are in inputs after from_json.
    input_refs = extract_refs_from_values(server_call.inputs)
    inputs = from_json(server_call.inputs, server_call.project_id, server)
    call = Call(
        op_name=server_call.op_name,
        project_id=server_call.project_id,
        trace_id=server_call.trace_id,
        parent_id=server_call.parent_id,
        id=server_call.id,
        inputs=inputs,
        input_refs=input_refs,
        output=output,
        summary=server_call.summary,
    )
    if call.id is None:
        raise ValueError("Call ID is None")
    return call


class CallsIter:
    server: TraceServerInterface
    filter: _CallsFilter

    def __init__(
        self, server: TraceServerInterface, project_id: str, filter: _CallsFilter
    ) -> None:
        self.server = server
        self.project_id = project_id
        self.filter = filter

    def __getitem__(self, key: Union[slice, int]) -> Call:
        if isinstance(key, slice):
            raise NotImplementedError("Slicing not supported")
        for i, call in enumerate(self):
            if i == key:
                return call
        raise IndexError(f"Index {key} out of range")

    def __iter__(self) -> Iterator[Call]:
        page_index = 0
        page_size = 1000
        entity, project = self.project_id.split("/")
        while True:
            response = self.server.calls_query(
                CallsQueryReq(
                    project_id=self.project_id,
                    filter=self.filter,
                    offset=page_index * page_size,
                    limit=page_size,
                )
            )
            page_data = response.calls
            for call in page_data:
                # TODO: if we want to be able to refer to call outputs
                # we need to yield a ref-tracking call here.
                yield make_client_call(entity, project, call, self.server)
                # yield make_trace_obj(call, ValRef(call.id), self.server, None)
            if len(page_data) < page_size:
                break
            page_index += 1


def weave_client_calls(self: WeaveClient, op_names, input_refs=None) -> CallsIter:
    trace_server_filt = _CallsFilter()
    if op_names:
        if isinstance(op_names, str):
            op_names = [op_names]
        op_ref_uris = []
        for op_name in op_names:
            if op_name.startswith("weave:///"):
                op_ref_uris.append(op_name)
            else:
                if ":" not in op_name:
                    op_name = op_name + ":*"
                op_ref_uris.append(f"weave:///{self._project_id()}/op/{op_name}")
        trace_server_filt.op_names = op_ref_uris
    if input_refs:
        trace_server_filt.input_refs = input_refs
    return CallsIter(self.server, self._project_id(), trace_server_filt)


def weave_client_ops(
    self: WeaveClient, filter: Optional[_ObjectVersionFilter] = None
) -> list[ObjSchema]:
    if not filter:
        filter = _ObjectVersionFilter()
    else:
        filter = filter.model_copy()
    filter = cast(_ObjectVersionFilter, filter)
    # TODO: fetches latest_only. Need to be more general.
    filter.latest_only = True
    filter.is_op = True

    response = self.server.objs_query(
        ObjQueryReq(
            project_id=self._project_id(),
            filter=filter,
        )
    )
    return response.objs


def weave_client_objs(
    self: WeaveClient, filter: Optional[_ObjectVersionFilter] = None, types=None
) -> list[ObjSchema]:
    if not filter:
        filter = _ObjectVersionFilter()
    else:
        filter = filter.model_copy()
    if types is not None:
        if isinstance(types, str):
            types = [types]
        filter.base_object_classes = types
    filter = cast(_ObjectVersionFilter, filter)
    # TODO: fetches latest_only. Need to be more general.
    # filter.latest_only = True
    filter.is_op = False

    response = self.server.objs_query(
        ObjQueryReq(
            project_id=self._project_id(),
            filter=filter,
        )
    )
    return response.objs


def weave_client_get_batch(self, refs: Sequence[str]) -> Sequence[Any]:
    # Create a dictionary to store unique refs and their results
    unique_refs = list(set(refs))
    read_res = self.server.refs_read_batch(
        RefsReadBatchReq(refs=[uri for uri in unique_refs])
    )

    # Create a mapping from ref to result
    ref_to_result = {
        unique_refs[i]: from_json(val, self._project_id(), self.server)
        for i, val in enumerate(read_res.vals)
    }

    # Return results in the original order of refs
    return [ref_to_result[ref] for ref in refs]
