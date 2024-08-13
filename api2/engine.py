from typing import Optional, Union, Any, Sequence
from weave.weave_client import WeaveClient
from weave.trace_server.trace_server_interface import (
    _CallsFilter,
    TraceServerInterface,
    CallsQueryReq,
    RefsReadBatchReq,
)

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from api2.proto import Query
from weave.weave_client import WeaveClient, from_json


from weave.trace.refs import CallRef
from api2 import engine_context
from api2.query_api import *
from weave.trace.vals import ObjectRecord


def unweaveify(val):
    if isinstance(val, list):
        return [unweaveify(v) for v in val]
    elif isinstance(val, dict):
        return {k: unweaveify(v) for k, v in val.items()}
    elif isinstance(val, ObjectRecord):
        return val.__dict__
    return val


@dataclass
class GroupbyAggLoc:
    gb_agg: "GroupbyAgg"

    def __getitem__(self, idx):
        gb_agg = self.gb_agg
        print("IDX", idx)
        return GroupbyAgg(
            gb_agg.dfgb.filter(lambda x: x.name in idx).groupby(
                gb_agg.dfgb.grouper.names
            ),
            gb_agg.agg.loc[idx],
        )


@dataclass
class GroupbyAgg:
    dfgb: DataFrameGroupBy
    agg: pd.DataFrame

    @property
    def loc(self):
        return GroupbyAggLoc(self)

    # def loc(self, idx):
    #     return GroupbyAgg(self.dfgb.filter(lambda x: x.name in idx), self.agg.loc[idx])


class Engine:
    server: TraceServerInterface

    def __init__(self, project_id, server):
        self.project_id = project_id
        self.server = server
        self._cache = {}

    def _calls_df(
        self,
        _filter: _CallsFilter,
        select: Optional[Union[str, list[str]]] = None,
        limit: Optional[int] = None,
    ):
        vals = []
        for page in self.calls_iter_pages(_filter, select=select, limit=limit):
            vals.extend(page)
        return pd.json_normalize(vals)

    def calls_iter_pages(
        self,
        _filter: _CallsFilter,
        select: Optional[Union[str, list[str]]] = None,
        limit: Optional[int] = None,
    ):
        page_index = 0
        page_size = 1000
        remaining = limit
        while True:
            response = self.server.calls_query(
                CallsQueryReq(
                    project_id=self.project_id,
                    filter=_filter,
                    offset=page_index * page_size,
                    limit=page_size,
                )
            )
            page_data = []
            for v in response.calls:
                v = v.model_dump()
                if select:
                    page_data.append({k: v[k] for k in select})
                else:
                    page_data.append(v)
            if remaining is not None:
                page_data = page_data[:remaining]
                remaining -= len(page_data)
            yield page_data
            if len(page_data) < page_size:
                break
            page_index += 1

    def expand_refs(self, refs: Sequence[str]) -> Sequence[Any]:
        # Create a dictionary to store unique refs and their results
        unique_refs = list(set(refs))
        read_res = self.server.refs_read_batch(
            RefsReadBatchReq(refs=[uri for uri in unique_refs])
        )

        # Create a mapping from ref to result
        ref_to_result = {
            unique_refs[i]: from_json(val, self.project_id, self.server)
            for i, val in enumerate(read_res.vals)
        }

        # Return results in the original order of refs
        return [ref_to_result[ref] for ref in refs]

    def execute(self, query: Query):
        op = query.from_op
        if op in self._cache:
            return self._cache[op]

        res = self._execute(query)
        self._cache[op] = res
        return res

    def _execute(self, query: Query):
        op = query.from_op
        if isinstance(op, DBOpCallsTableRoot):
            print("OP:", type(op))
            vals = []
            for i, page in enumerate(
                self.calls_iter_pages(query._filter, limit=query.limit)
            ):
                print("page i", i)
                vals.extend(page)
            df = pd.json_normalize(vals)
            if df.empty:
                return df
            # df = pd_apply_and_insert(df, "op_name", query.split_obj_ref)
            entity, project = self.project_id.split("/", 1)
            call_refs = [CallRef(entity, project, c["id"]).uri() for c in vals]
            df.index = pd.Index(call_refs)
            print("OP end:", type(op))
            return df
        elif isinstance(op, DBOpLen):
            print("OP:", type(op))
            df = self.execute(op.input)
            print("OP end:", type(op))
            return len(df)
        elif isinstance(op, DBOpCallsChildren):
            print("OP:", type(op))
            calls = self.execute(op.input)
            if calls.empty:
                return calls
            parent_ids = list(calls["id"])
            print("parents", len(parent_ids))
            filt = _CallsFilter(parent_ids=parent_ids)
            vals = []
            for i, page in enumerate(self.calls_iter_pages(filt)):
                print("page", i)
                vals.extend(page)
            children = pd.json_normalize(vals)
            if children.empty:
                return children
            calls = calls.reset_index()
            merged_df = children.merge(
                calls[["id", "index"]],
                left_on="parent_id",
                right_on="id",
                suffixes=("", ".orig_parent"),
            )
            merged_df["rank"] = merged_df.groupby("parent_id").cumcount() + 1
            children_with_index = merged_df.set_index(["index", "rank"])
            children_with_index.index.names = ["parent_ref", "rank"]

            print("OP end:", type(op))
            return children_with_index

        elif isinstance(op, DBOpCounts):
            print("OP:", type(op))
            df = self.execute(op.input)
            if df.empty:
                return df
            counts = df.groupby(level="parent_ref").size()
            print("OP end:", type(op))
            return counts
        elif isinstance(op, DBOpNth):
            print("OP:", type(op))
            df = self.execute(op.input)
            if df.empty:
                return df
            nth = df.groupby(level="parent_ref").nth(op.idx)
            print("OP end:", type(op))
            return nth.reset_index(level="rank")
        elif isinstance(op, DBOpCallsColumn):
            df = self.execute(op.input)
            if df.empty:
                return df
            return df[op.col_name]
        elif isinstance(op, DBOpCallsColumns):
            df = self.execute(op.input)
            if df.empty:
                return df

            # TODO: This should probably return column types, using something
            # like query.friendly_dtypes
            return df.columns
        # elif isinstance(op, DBOpMap):
        #     df = self.execute(op.input)
        #     if df.empty:
        #         return df

        #     # Bridge back to WeaveMap for now
        #     local_df = LocalDataframe(df)
        #     map = WeaveMap(local_df, OpCall(op.op, op.column_mapping))
        #     return map.get_result()
        elif isinstance(op, DBOpGroupBy):
            df = self.execute(op.input)
            if df.empty:
                return df
            return df.groupby(op.col_name)
        elif isinstance(op, DBOpCallsGroupbyGroups):
            # Don't do this!
            gb_agg = self.execute(op.input)
            dfgb = gb_agg.dfgb
            result = dfgb.apply(lambda x: x)
            result.index = result.index.rename("ref", level=-1)
            return result
        elif isinstance(op, DBOpAgg):
            dfgb = self.execute(op.input)
            agged = dfgb.agg(op.agg)
            agged.columns = [".".join(col) for col in agged.columns]
            agged[agged.index.name] = agged.index
            return GroupbyAgg(dfgb, agged)
        elif isinstance(op, DBOpLoc):
            df = self.execute(op.input)
            return df.loc[op.idx]
        elif isinstance(op, DBOpExpandRef):
            df = self.execute(op.input)
            objs = self.expand_refs(df)
            unweaved_objs = unweaveify(objs)
            obj_df = pd.json_normalize(unweaved_objs)
            obj_df.index = df
            obj_df.index.name = "ref"
            return obj_df
        else:
            raise ValueError(f"Unhandled op {op}")

        # if isinstance(query.from_op,
        # print("Query", query)


def init_engine(wc: WeaveClient):
    engine_context.ENGINE = Engine(wc._project_id(), wc.server)
