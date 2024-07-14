import inspect
from typing import cast, Optional, Union, Callable, Any
import dataclasses
from dataclasses import dataclass

import pandas as pd

from api2.proto import Executable, QuerySequence, OpType
from api2.engine_context import get_engine
from api2.provider import make_source, SourceType


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
