import inspect
from typing import cast, Optional, Callable, Any, Protocol, TypeVar
from collections.abc import KeysView, ValuesView, ItemsView

import dataclasses
from dataclasses import dataclass, field

import pandas as pd

from weave import Dataset
from api2.proto import OpType
from api2.cache import batch_get, batch_fill, NOT_COMPUTED

K = TypeVar("K")
V = TypeVar("V")


class DictLike(Protocol[K, V]):
    def __getitem__(self, key: K) -> V: ...

    # def __setitem__(self, key: K, value: V) -> None: ...
    # def __delitem__(self, key: K) -> None: ...
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]: ...
    def keys(self) -> "KeysView[K]": ...
    def values(self) -> "ValuesView[V]": ...
    def items(self) -> "ItemsView[K, V]": ...


def get_op_param_names(op: Callable):
    sig = inspect.signature(op)
    return list(sig.parameters.keys())


@dataclass
class OpCall:
    op: Callable
    inputs: Optional[dict[str, str]] = dataclasses.field(default_factory=dict)

    def input_names(self):
        return get_op_param_names(self.op)


@dataclass
class Pipeline:
    steps: dict[str, list[OpCall]] = dataclasses.field(default_factory=dict)

    def add_step(
        self,
        op: Callable,
        column_mapping: Optional[dict[str, str]] = dataclasses.field(
            default_factory=dict
        ),
        step_id: Optional[str] = None,
    ):
        # Cast for now.
        op = cast(OpType, op)
        if step_id is None:
            step_id = op.name
        op_call = OpCall(op, column_mapping)
        if step_id not in self.steps:
            self.steps[step_id] = []
        self.steps[step_id].append(op_call)
        return op_call

    def free_inputs(self) -> list[str]:
        free_inputs = []
        avail_vars = set()
        for step_id, op_calls in self.steps.items():
            for op_call in op_calls:
                for name in op_call.input_names():
                    if name not in avail_vars:
                        free_inputs.append(name)

            # Append "_output" for now.
            avail_vars.add(step_id + "_output")
        return free_inputs

    def lazy_call(self, params: DictLike, n_trials: int = 1) -> "BoundPipeline":
        params = to_dict_like(params)
        missing_inputs = set(self.free_inputs()).difference(params.keys())
        if missing_inputs:
            raise ValueError(f"Missing inputs: {missing_inputs}")
        # TODO: check avail keys on params populate free_inputs before constructing
        # BoundPipeline
        return BoundPipeline(params, self, n_trials)


@dataclass
class BoundPipeline:
    params: DictLike[str, Any]
    pipeline: Pipeline
    n_trials: int = 1

    def fetch_existing_results(self) -> "PipelineResults":
        return PipelineResults(self)


class DictLikeDataset(DictLike):
    def __init__(self, dataset: Dataset):
        self.rows = list(dataset.rows)

    def keys(self):
        return self.rows[0].keys()

    def __getitem__(self, key):
        return [row[key] for row in self.rows]

    # TODO: missing methods


def to_dict_like(v) -> DictLike:
    if isinstance(v, Dataset):
        return DictLikeDataset(v)
    elif isinstance(v, dict):
        return v
    else:
        raise ValueError(f"Unsupported type {type(v)}")


@dataclass
class OpCallResults:
    params: pd.DataFrame
    result: pd.Series


@dataclass
class PipelineResults:
    bound_pipeline: BoundPipeline
    results: dict[str, dict[str, pd.Series]] = field(init=False)

    def __post_init__(self):
        self.results = {}
        initial_df = pd.DataFrame(dict(self.bound_pipeline.params))
        if self.bound_pipeline.n_trials > 1:
            dfs = []
            for i in range(0, self.bound_pipeline.n_trials):
                df = initial_df.copy()
                # TODO: this is not yet used in weave caching. Need to build
                # in!
                df["weave_trial"] = i
                dfs.append(df)
            initial_df = pd.concat(dfs)
        self._initial_df = initial_df
        # self._initial_df = pd.concat(
        #     [self._initial_df] * self.bound_pipeline.n_trials
        # )
        self.fill_from_cache()

    def _process_pipeline(self, fill_from_cache=False):
        t = self._initial_df.copy()
        for step_id, op_calls in self.bound_pipeline.pipeline.steps.items():
            t = self._process_step(step_id, op_calls, t, fill_from_cache)
            t.index = range(len(t))
        return t

    def _process_step(self, step_id, op_calls, t, fill_from_cache):
        op_call_results = []
        for op_call in op_calls:
            if fill_from_cache:
                result = batch_get(t, op_call.op)
                self.results.setdefault(step_id, {})[op_call.op.ref.uri()] = result
            else:
                result = self.results[step_id][op_call.op.ref.uri()]

            op_call_columns = pd.DataFrame(
                {
                    f"{step_id}_op_ref": [op_call.op.ref.uri()] * len(result),
                    f"{step_id}_output": result,
                }
            )
            op_call_result = pd.concat([t, op_call_columns], axis=1)
            op_call_results.append(op_call_result)
        return pd.concat(op_call_results)

    def fill_from_cache(self):
        self._process_pipeline(fill_from_cache=True)

    def execute_cost(self):
        to_compute = {
            op_uri: (op_call_result == NOT_COMPUTED).sum()
            for step_id, op_call_results in self.results.items()
            for op_uri, op_call_result in op_call_results.items()
        }
        return {"to_compute": to_compute}

    def execute(self):
        t = self._initial_df.copy()
        for step_id, op_calls in self.bound_pipeline.pipeline.steps.items():
            for op_call in op_calls:
                result = self.results[step_id][op_call.op.ref.uri()]
                yield from batch_fill(t, op_call.op, result)
            t = self._process_step(step_id, op_calls, t, fill_from_cache=False)
            t.index = range(len(t))

    def to_pandas(self):
        return self._process_pipeline(fill_from_cache=False)

    def summary(self):
        # This would do summarization like we do for evals currently, but presumably
        # be more configurable.
        raise NotImplemented
