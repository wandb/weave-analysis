from typing import cast, Optional, Union, Any
from weave.weave_client import WeaveClient
from weave.trace_server.trace_server_interface import (
    _CallsFilter,
    TraceServerInterface,
    CallsQueryReq,
)

from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
import weave
import query
from api2.proto import Query
from pandas_util import get_unflat_value

from weave.trace.refs import CallRef
from api2.pipeline import WeaveMap
from api2 import engine_context
from api2.provider import *


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
        # OK I'm here in this function, working on making it handle
        # trials.
        # Its not working yet, and something seems slow.
        # Also is it just better to log a trial number in call? Well
        # we can't really because of being distributed, don't want a global
        # order key.

        # TODO: not ideal, uses client, not server, but engine
        # should be able to run on server
        op_ref = weave.publish(map.op_call.op)

        # Currently loads all historical calls of op.
        calls_df = self._calls_df(_CallsFilter(op_names=[op_ref.uri()]))

        find_inputs_df = self._map_calls_params(map)
        result_index = pd.MultiIndex.from_product(
            [find_inputs_df.index, list(range(map.n_trials))]
        )

        if not len(calls_df):
            return pd.Series([NOT_COMPUTED] * len(result_index), index=result_index)

        calls_df_norm = calls_df.astype(str)
        calls_df_norm = calls_df_norm.reset_index(names="calls_index")

        # Prepare the op_configs DataFrame
        # op_configs_df_normalized = pd.json_normalize(
        #     find_inputs_df.to_dict(orient="records"), sep="."
        # )
        find_inputs_df_normalized = find_inputs_df.add_prefix("inputs.")
        find_inputs_df_normalized = find_inputs_df_normalized.astype(str)
        find_inputs_df_normalized = find_inputs_df_normalized.reset_index(
            names="inputs_index"
        )

        # Perform the merge operation
        merged_df = calls_df_norm.merge(find_inputs_df_normalized, how="right")

        # Group by the original index and aggregate the results
        grouped = merged_df.groupby("inputs_index")

        # Initialize results list with NOT_COMPUTED for all op_configs

        results = {}
        for index, group in grouped:
            # if not isinstance(index, int):
            #     # This is to make the type checker happy. We know its int
            #     # because we constructed it above.
            #     raise ValueError(f"Expected index to be an int, got {index}")
            # If there's at least one match, use the first match
            # call_index = match_row["calls_index"]
            for i, (_, row) in enumerate(group.iterrows()):
                import math

                if not math.isnan(row["calls_index"]):
                    orig_call = calls_df.loc[row["calls_index"]]
                    if i < map.n_trials:
                        results[(index, i)] = get_unflat_value(orig_call, "output")

        return pd.Series(results, index=result_index, name="output")

    def map_cost(self, map: "WeaveMap"):
        cached_results = self._map_cached_results(map)
        return {"to_compute": sum([1 for r in cached_results if r is NOT_COMPUTED])}

    def map_execute(self, map: "WeaveMap"):
        work_to_do = {}
        op_configs = self._map_calls_params(map)
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
        else:
            raise ValueError(f"Unhandled op {op}")

        # if isinstance(query.from_op,
        # print("Query", query)


def init_engine(wc: WeaveClient):
    if engine_context.ENGINE:
        return
    engine_context.ENGINE = Engine(wc._project_id(), wc.server)
