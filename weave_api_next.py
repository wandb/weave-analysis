# This contains code that needs to be added to the weave Python package.
# But its here for now so we can iterate on finding the right patterns.

from typing import cast, Optional, Union, Iterator, Sequence, Any
from weave.weave_client import WeaveClient, Call, from_json
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
from weave.trace.vals import TraceObject


def make_client_call(
    entity: str, project: str, server_call: CallSchema, server: TraceServerInterface
) -> Call:
    output = server_call.output
    call = Call(
        op_name=server_call.op_name,
        project_id=server_call.project_id,
        trace_id=server_call.trace_id,
        parent_id=server_call.parent_id,
        id=server_call.id,
        inputs=from_json(server_call.inputs, server_call.project_id, server),
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


def weave_client_calls(self: WeaveClient, op_names) -> CallsIter:
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
    trace_server_filt = _CallsFilter(op_names=op_ref_uris)
    return CallsIter(self.server, self._project_id(), trace_server_filt)


def weave_client_ops(
    self: WeaveClient, filter: Optional[_ObjectVersionFilter] = None
) -> list[ObjSchema]:
    if not filter:
        filter = _ObjectVersionFilter()
    else:
        filter = filter.model_copy()
    filter = cast(_ObjectVersionFilter, filter)
    filter.is_op = True

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
