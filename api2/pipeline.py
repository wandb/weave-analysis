import inspect
from typing import cast, Optional, Union, Callable, Any, Mapping, Protocol, TypeVar
from collections.abc import KeysView, ValuesView, ItemsView

import dataclasses
from dataclasses import dataclass, field

import pandas as pd

from weave import Dataset
from api2.proto import Executable, QuerySequence, OpType
from api2.engine_context import get_engine
from api2.provider import make_source, SourceType
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
class WeaveMap(Executable):
    data: QuerySequence
    op_call: "OpCall"
    n_trials: int = 1

    result: Optional[pd.Series] = None

    @property
    def column_mapping(self):
        if not self.op_call.inputs:
            return {k: k for k in get_op_param_names(self.op_call.op)}
        return self.op_call.inputs

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

    def get_result(self):
        for delta in self.execute():
            pass
        return self.result


def weave_map(
    data: SourceType,
    op: Callable,
    inputs: Optional[dict[str, str]] = None,
    n_trials: int = 1,
):
    return WeaveMap(make_source(data), OpCall(op, inputs), n_trials)


@dataclass
class OpCall:
    op: Callable
    inputs: Optional[dict[str, str]] = dataclasses.field(default_factory=dict)

    def input_names(self):
        return get_op_param_names(self.op)


@dataclass
class PipelineStep:
    op_call: OpCall
    step_id: str


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
        # Cast for now.
        op = cast(OpType, op)
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
                map = WeaveMap(self.base, step.op_call)
                results[step.step_id] = map.cost()
            elif isinstance(step, PipelineStepCompare):
                raise NotImplemented
        return results

    def execute(self):
        results = {}
        for step in self.steps.values():
            if isinstance(step, PipelineStep):
                map = WeaveMap(self.base, step.op_call)
                for delta in map.execute():
                    yield {"step": step.step_id, **delta}
                results[step.step_id] = map.result
            elif isinstance(step, PipelineStepCompare):
                raise NotImplemented
        self.result = pd.DataFrame(results)

    def get_result(self):
        for delta in self.execute():
            pass
        return cast(pd.DataFrame, self.result)


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
        self._initial_df = pd.DataFrame(dict(self.bound_pipeline.params))
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
