import inspect
from typing import Optional, Union, Callable, Any, Iterator, Protocol
import dataclasses
from dataclasses import dataclass
from weave.weave_client import WeaveClient
from weave.trace.refs import CallRef
import glob
from weave.trace_server.trace_server_interface import (
    _CallsFilter,
    TraceServerInterface,
    CallsQueryReq,
)
import openai

from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import weave
import query
from weave.client_context.weave_client import require_weave_client
from pandas_util import pd_apply_and_insert, get_unflat_value


class Executable(Protocol):
    def cost(self) -> dict: ...

    def execute(self) -> Iterator[dict]: ...


def get_op_param_names(op: Callable):
    sig = inspect.signature(op)
    return list(sig.parameters.keys())


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
        for page in get_engine().calls_iter_pages(_filter, select=select, limit=limit):
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
        input_df = map.data.to_pandas()
        return pd.DataFrame(
            {
                op_key: input_df[data_key]
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
        results = [NOT_COMPUTED] * len(op_configs_df)

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
            op_config = op_configs.loc[i]
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


ENGINE: Engine


def get_engine() -> Engine:
    return ENGINE


@dataclass
class Index:
    level_names: list[str]
    values: list[tuple]


@dataclass
class CallsQuery:
    entity: str
    project: str
    _filter: _CallsFilter
    limit: Optional[int] = None

    # def filter(self, column, value):
    #     return CallsQuery(self.df[self.df[column] == value])

    def groupby(self, column):
        pass

    def columns(self):
        return CallsColumnsQuery(self)

    def column(self, column):
        return CallsQueryColumn(self, column)

    def __len__(self):
        engine = get_engine()
        return engine.calls_len(self._filter, limit=self.limit)

    def to_pandas(self) -> pd.DataFrame:
        vals = []
        for page in get_engine().calls_iter_pages(self._filter, limit=self.limit):
            vals.extend(page)
        df = pd.json_normalize(vals)
        if df.empty:
            return df
        # df = pd_apply_and_insert(df, "op_name", query.split_obj_ref)
        call_refs = [CallRef(self.entity, self.project, c["id"]).uri() for c in vals]
        df.index = call_refs
        return df


class CallsQueryGroupby:
    calls_query: "CallsQuery"
    by: "CallsQueryColumn"

    def column(self):
        pass


@dataclass
class CallsQueryColumn:
    calls_query: "CallsQuery"
    column_name: str

    def __len__(self):
        return len(self.calls_query)

    def to_pandas(self) -> pd.Series:
        # TODO: This logic should be shared with CallsQuery.to_pandas
        vals = []
        for page in get_engine().calls_iter_pages(
            self.calls_query._filter, limit=self.calls_query.limit
        ):
            vals.extend(page)
        df = pd.json_normalize(vals)
        if df.empty:
            return pd.Series()
        call_refs = [
            CallRef(self.calls_query.entity, self.calls_query.project, c["id"]).uri()
            for c in vals
        ]
        df.index = call_refs
        return df[self.column_name]


@dataclass
class CallsColumnsQuery:
    calls_query: CallsQuery
    column_filter = None  # TODO

    def __iter__(self):
        engine = get_engine()
        columns = engine.calls_columns(
            self.calls_query._filter, limit=self.calls_query.limit
        )
        for k, v in columns.items():
            yield k, v

    def __str__(self):
        return str(dict(self))


def calls(
    self: WeaveClient,
    op_names: Optional[Union[str, list[str]]],
    limit: Optional[int] = None,
) -> CallsQuery:
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
    return CallsQuery(self.entity, self.project, trace_server_filt, limit=limit)


@dataclass
class WeaveMap(Executable):
    data: Any
    op_call: "OpCall"
    n_trials: Optional[int]

    result: Optional[pd.Series] = None

    @property
    def column_mapping(self):
        if not self.op_call.column_mapping:
            return {k: k for k in get_op_param_names(self.op_call.op)}
        return self.op_call.column_mapping

    def cost(self):
        engine = get_engine()
        return engine.map_cost(self)

    def execute(self):
        engine = get_engine()
        results = []
        for item_result in engine.map_execute(self):
            results.append(item_result)
            yield item_result
        self.result = pd.Series(
            [r["val"] for r in results],
            index=[r["index"] for r in results],
        )


def weave_map(
    data,
    op_call: "OpCall",
    n_trials: Optional[int] = None,
):
    return WeaveMap(data, op_call, n_trials)


class QuerySequence(Protocol):
    pass


@dataclass
class OpCall:
    op: Any
    column_mapping: Optional[dict[str, str]] = dataclasses.field(default_factory=dict)


@dataclass
class PipelineStep:
    op_call: OpCall
    step_id: str
    n_trials: int


@dataclass
class PipelineStepCompare:
    op_calls: list[OpCall]
    step_id: str
    n_trials: int

    def add_op(self, op: Any, column_mapping: Optional[dict[str, str]] = None):
        if column_mapping is None:
            column_mapping = {k: k for k in get_op_param_names(op)}
        self.op_calls.append(OpCall(op, column_mapping))


@dataclass
class BatchPipeline(Executable):
    base: QuerySequence
    steps: dict[str, Union[PipelineStep, PipelineStepCompare]] = dataclasses.field(
        default_factory=dict
    )
    result: Optional[pd.DataFrame] = None

    def add_step(
        self,
        op: Callable,
        column_mapping: Optional[dict[str, str]] = dataclasses.field(
            default_factory=dict
        ),
        step_id: Optional[str] = None,
        n_trials: Optional[int] = None,
    ):
        if step_id is None:
            step_id = op.name
        if n_trials is None:
            n_trials = 1
        step = PipelineStep(OpCall(op, column_mapping), step_id, n_trials)
        self.steps[step_id] = step
        return step

    def add_compare_step(self, step_name: str, n_trials: Optional[int] = None):
        # This forks the pipeline
        if n_trials is None:
            n_trials = 1
        step = PipelineStepCompare([], step_name, n_trials)
        self.steps[step_name] = step
        return step

    def cost(self):
        results = {}
        for step in self.steps.values():
            if isinstance(step, PipelineStep):
                map = weave_map(self.base, step.op_call)
                results[step.step_id] = map.cost()
            elif isinstance(step, PipelineStepCompare):
                raise NotImplemented
        return results

    def execute(self):
        results = {}
        for step in self.steps.values():
            if isinstance(step, PipelineStep):
                map = weave_map(self.base, step.op_call)
                for delta in map.execute():
                    yield {"step": step.step_id, **delta}
                results[step.step_id] = map.result
            elif isinstance(step, PipelineStepCompare):
                raise NotImplemented
        self.result = pd.DataFrame(results)


@weave.op()
def summarize_purpose(code: str):
    prompt = f"Summarize the purpose of the following code, in 5 words or less.\n\n{code}\n\nPurpose:"
    response = openai.chat.completions.create(
        model="gpt-4o", messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content


@weave.op()
def classify(code: str):
    prompt = f"Classify the following code. Return a single word.\n\n{code}\n\nClass:"
    response = openai.chat.completions.create(
        model="gpt-4o", messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content


@weave.op()
def strlen(s: str):
    return len(s)


wc = weave.init_local_client("testableapi.db")
ENGINE = Engine(wc._project_id(), wc.server)


class LocalQueryable(QuerySequence):
    def __init__(self, df):
        self.df = df

    def to_pandas(self):
        return self.df


def summarize_curdir_py():
    files = []
    for f_name in glob.glob("*.py"):
        files.append({"name": f_name, "code": open(f_name).read()})
    map = weave_map(
        LocalQueryable(pd.DataFrame(files)), OpCall(summarize_purpose), n_trials=1
    )
    print(map.cost())
    for delta in map.execute():
        pass
    return map.result


if __name__ == "__main__":
    # Create data
    print("SUMMARIZE RESULT", summarize_curdir_py())

    # # Read
    my_calls = calls(wc, "summarize_purpose", limit=100)
    print("# Calls", len(my_calls))
    print("Columns", my_calls.columns())
    # my_calls.append_column(classify, my_calls.column("inputs.code"))
    # my_calls.append_column(strlen, my_calls.column("output"))
    # my_calls.cost()
    # grouped = my_calls.groupby("classify")
    # grouped.add_column("mean", grouped.column('strlen')
    # print('grouped cost', grouped.cost())
    # print(grouped)

    # input_code_class = weave_map(my_calls, classify, {"code": "inputs.code"})
    # print("Classify cost", input_code_class.cost())
    # for result in input_code_class.execute():
    #     pass

    # output_len = weave_map(my_calls, strlen, {"s": "output"})
    # print("Strlen cost", input_code_class.cost())
    # for result in output_len.execute():
    #     pass

    # both = pd.concat([input_code_class.result, output_len.result], axis=1)
    # print(both.groupby("classify").mean())

    # TODO: get this grouping down into API
    # show how to fetch all calls in a group. (streamlit example)
    # show fetching up child calls
    # show working
    # Make work into pivot table
    # index (input_ref, trial), columns: one level for each branch

    p = BatchPipeline(my_calls)
    p.add_step(classify, column_mapping={"code": "inputs.code"})
    p.add_step(strlen, column_mapping={"s": "output"})
    print(p.cost())
    for delta in p.execute():
        pass
    print(p.result.groupby("classify").mean())

    # TODO:
    #   - make compare_step work
    #   - make n_trials work
    #   - better argument passing (instead of column_mapping?)
    #   - non-batch version
    #   - make work on raw data?
    #   - better error handling
    #   - I need whole Call information available, not just output
    #     (for example, for building a nice Eval table with feedback)

    # res = p.execute()
    # print(res)

    # This is how BatchPipeline would work for an Eval

    # eval = eval_picker()
    # ds = eval.dataset
    # models = model_picker()
    # p = BatchPipeline(ds)
    # model_step = p.add_compare_step(step_name="model", n_trials=eval.n_trials)
    # for model in models:
    #     model_step.add_op(model.op)
    # for score_fn in eval.score_fns:
    #     p.add_step(score_fn.op, {})

    # print(t.cost())
    # for result_delta in t.execute():
    #     print(result_delta)
    # df = t.result.dataframe()
