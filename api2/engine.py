from typing import cast, Optional, Union, Any, Sequence
from dataclasses import fields
from weave.weave_client import WeaveClient
from weave.trace_server.trace_server_interface import (
    _CallsFilter,
    TraceServerInterface,
    CallsQueryReq,
    RefsReadBatchReq,
)

from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
import weave
import query
from api2.proto import Query
from pandas_util import get_unflat_value
from weave.weave_client import WeaveClient, from_json

from weave_api_next import weave_client_get_batch

from weave.trace.refs import CallRef
from api2.pipeline import WeaveMap, OpCall
from api2 import engine_context
from api2.provider import *
from weave.trace.vals import ObjectRecord


def unweaveify(val):
    if isinstance(val, list):
        return [unweaveify(v) for v in val]
    elif isinstance(val, dict):
        return {k: unweaveify(v) for k, v in val.items()}
    elif isinstance(val, ObjectRecord):
        return val.__dict__
    return val


class NotComputed:
    def __repr__(self):
        return "<NotComputed>"


NOT_COMPUTED = NotComputed()


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


def sum_dict_leaves(dict1, dict2):
    def merge_sum(d1, d2):
        result = {}
        all_keys = set(d1.keys()) | set(d2.keys())

        for key in all_keys:
            if key in d1 and key in d2:
                if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    result[key] = merge_sum(d1[key], d2[key])
                elif isinstance(d1[key], (int, float)) and isinstance(
                    d2[key], (int, float)
                ):
                    result[key] = d1[key] + d2[key]
                else:
                    result[key] = d1[
                        key
                    ]  # In case of type mismatch, keep the value from dict1
            elif key in d1:
                result[key] = d1[key]
            else:
                result[key] = d2[key]

        return result

    return merge_sum(dict1, dict2)


def merge_cost(c1: dict, c2: dict):
    return sum_dict_leaves(c1, c2)


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

    def calls_columns(
        self,
        _filter: _CallsFilter,
        select: Optional[Union[str, list[str]]] = None,
        limit: Optional[int] = None,
    ):
        df = self._calls_df(_filter, select=select, limit=limit)
        return query.friendly_dtypes(df).to_dict()

    def calls_len(self, _filter: _CallsFilter, limit: Optional[int] = None):
        count = 0
        for page in self.calls_iter_pages(_filter, select=["id"], limit=limit):
            count += len(page)
        return count

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

    def _map_calls_params(self, map: "WeaveMap"):
        return pd.DataFrame(
            {
                op_key: map.data.column(data_key).to_pandas()
                for op_key, data_key in map.column_mapping.items()
            }
        )

    # How to get cached results?
    # we have all the inputs we want to find
    # load calls with those inputs
    # join to our input table, just to get original indexes
    # add a rank column (count original index)
    def _map_cached_results(self, map: "WeaveMap"):
        # TODO: not ideal, uses client, not server, but engine
        # should be able to run on server
        op_ref = weave.publish(map.op_call.op)

        calls_df = self._calls_df(_CallsFilter(op_names=[op_ref.uri()]))

        # Convert op_configs to a DataFrame
        op_configs_df = self._map_calls_params(map)
        results: list[Any] = [NOT_COMPUTED] * len(op_configs_df)

        if not len(calls_df):
            return pd.Series(results, index=op_configs_df.index)

        # Add an index column to keep track of the original order
        op_configs_df["original_index"] = range(len(op_configs_df))

        # Prepare the calls DataFrame
        calls_df_norm = calls_df.astype(str)
        calls_df_norm["call_index"] = calls_df.index

        # Prepare the op_configs DataFrame
        op_configs_df_normalized = pd.json_normalize(
            op_configs_df.to_dict(orient="records"), sep="."
        )
        op_configs_df_normalized = op_configs_df_normalized.add_prefix("inputs.")
        op_configs_df_normalized = op_configs_df_normalized.astype(str)

        # Perform the merge operation
        merged_df = calls_df_norm.merge(
            op_configs_df_normalized, how="right", indicator=True
        )

        # Group by the original index and aggregate the results
        grouped = merged_df.groupby("inputs.original_index")

        # Initialize results list with NOT_COMPUTED for all op_configs

        for index, group in grouped:
            # if not isinstance(index, int):
            #     # This is to make the type checker happy. We know its int
            #     # because we constructed it above.
            #     raise ValueError(f"Expected index to be an int, got {index}")
            if "both" in group["_merge"].values:
                # If there's at least one match, use the first match
                match_row = group[group["_merge"] == "both"].iloc[0]
                call_index = match_row["call_index"]
                results[int(index)] = get_unflat_value(
                    calls_df.loc[call_index], "output"
                )

        return pd.Series(results, index=op_configs_df.index)

    def map_cost(self, map: "WeaveMap"):
        cached_results = self._map_cached_results(map)
        return {"to_compute": sum([1 for r in cached_results if r is NOT_COMPUTED])}

    def map_execute(self, map: "WeaveMap"):
        work_to_do = {}
        op_configs = self._map_calls_params(map)
        # Remove dupes, but we still do too much work.
        # TODO: ensure we only work on each unique value once!
        op_configs = op_configs[~op_configs.index.duplicated()]

        results = self._map_cached_results(map)

        for i, result in results.items():
            value = result
            if value is not NOT_COMPUTED:
                yield {"index": i, "val": value}
                continue
            # Don't know how to make the type checker happy here.
            op_config = op_configs.loc[i]  # type: ignore
            work_to_do[i] = op_config

        with ThreadPoolExecutor(max_workers=16) as executor:

            def do_one(work):
                i, op_config = work
                # try:
                return i, map.op_call.op(**op_config)
                # except:
                #     return i, None

            for index, result in executor.map(do_one, work_to_do.items()):
                yield {"index": index, "val": result}

    def cost(self, query: Query):
        return self._cost(query)

    def _cost(self, query: Query):
        op = query.from_op
        if isinstance(op, DBOpMap):
            input_cost = self.cost(op.input)
            # TODO: bad executes whole input
            input_df = self.execute(op.input)

            # Bridge back to WeaveMap for now
            local_df = LocalDataframe(input_df)
            map = WeaveMap(local_df, OpCall(op.op, op.column_mapping))
            return merge_cost(input_cost, map.cost())

        # if isinstance(op, DBOpMap):
        #     input_cost = self.cost(op.input)
        #     local_df = LocalDataframe(df)
        #     map = WeaveMap(local_df, OpCall(op.op, op.column_mapping))
        #     return merge_cost(input_cost, map.cost())
        # fields(op)

        # Cost algorithm
        # - first we must fill the whole query from cache.
        # - anything downstream of a cache miss is a miss, even if it
        #   would be a hit if we had the value
        # - the cost of filling the cache is essentially the cost of the
        #   whole query.
        #   - so is it worth it to allow map/cache inside this Query structure?
        #   - or should it be outside?
        #   - I think there's another
        # - yeah, my big question here is, is there 1 API, or 2? (query + batch)

        return {}

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
            return df.columns
        elif isinstance(op, DBOpMap):
            df = self.execute(op.input)
            if df.empty:
                return df

            # Bridge back to WeaveMap for now
            local_df = LocalDataframe(df)
            map = WeaveMap(local_df, OpCall(op.op, op.column_mapping))
            return map.get_result()
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
    if engine_context.ENGINE:
        return
    engine_context.ENGINE = Engine(wc._project_id(), wc.server)
