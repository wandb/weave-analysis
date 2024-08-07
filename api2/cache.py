from typing import Callable, Any
import pandas as pd
from weave.trace_server.trace_server_interface import _CallsFilter


import weave

from pandas_util import get_unflat_value

from api2.engine_context import get_engine
from api2.op_util import get_op_param_names
from concurrent.futures import ThreadPoolExecutor


class NotComputed:
    def __repr__(self):
        return "<NotComputed>"


NOT_COMPUTED = NotComputed()


def batch_get(params: pd.DataFrame, op: Callable):
    """Given a dataframe of op parameters, return existing results."""
    engine = get_engine()
    # TODO: not ideal, uses client, not server, but engine
    # should be able to run on server
    op_ref = weave.publish(op)

    op_param_names = get_op_param_names(op)
    missing_cols = set(op_param_names) - set(params.columns)
    if missing_cols:
        raise ValueError(
            f"Op expects parameters: {op_param_names}, but params contains: {params.columns}"
        )

    calls_df = engine._calls_df(_CallsFilter(op_names=[op_ref.uri()]))

    # Convert op_configs to a DataFrame

    op_configs_df = params.assign(original_index=range(len(params)))
    results: list[Any] = [NOT_COMPUTED] * len(op_configs_df)

    if not len(calls_df):
        return pd.Series(results, index=op_configs_df.index)

    # Add an index column to keep track of the original order
    # op_configs_df["original_index"] = range(len(op_configs_df))

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
            results[int(index)] = get_unflat_value(calls_df.loc[call_index], "output")

    return pd.Series(results, index=op_configs_df.index)


def batch_fill(params: pd.DataFrame, op: Callable, cache_result: pd.Series):
    work_to_do = {}
    params = params
    # Remove dupes, but we still do too much work.
    # TODO: ensure we only work on each unique value once!
    params = params[~params.index.duplicated()]

    op_param_names = get_op_param_names(op)
    params = params[op_param_names]

    results = cache_result

    for index, result in results.items():
        if result is not NOT_COMPUTED:
            continue
        # Don't know how to make the type checker happy here.
        work_to_do[index] = params.loc[index]  # type: ignore

    with ThreadPoolExecutor(max_workers=16) as executor:

        def do_one(work):
            index, op_params = work
            result = op(**op_params)
            # try:
            return index, result
            # except:
            #     return i, None

        for index, result in executor.map(do_one, work_to_do.items()):
            cache_result[index] = result
            yield {"index": index, "val": result}
