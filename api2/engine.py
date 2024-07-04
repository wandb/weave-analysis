from typing import cast, Optional, Union, Any
from weave.weave_client import WeaveClient
from weave.trace_server.trace_server_interface import (
    _CallsFilter,
    TraceServerInterface,
    CallsQueryReq,
)

from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import weave
import query
from pandas_util import get_unflat_value

from api2.pipeline import WeaveMap
from api2 import engine_context


class NotComputed:
    def __repr__(self):
        return "<NotComputed>"


NOT_COMPUTED = NotComputed()


class Engine:
    server: TraceServerInterface

    def __init__(self, project_id, server):
        self.project_id = project_id
        self.server = server

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


def init_engine(wc: WeaveClient):
    engine_context.ENGINE = Engine(wc._project_id(), wc.server)
