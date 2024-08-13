from dataclasses import dataclass
from typing import Any, Iterator, Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class DBOp:
    pass


class Executable(Protocol):
    def cost(self) -> dict: ...

    def execute(self) -> Iterator[dict]: ...

    def get_result(self) -> Any: ...


# Really this is like a Pandas series
# it has a name (which could be a tuple?) and an index.
# So spec this out more.
class Column(Protocol):
    def to_pandas(self) -> pd.Series: ...


@runtime_checkable
class Query(Protocol):
    from_op: DBOp


class ListsQuery(Query):
    pass


class OpType(Protocol):
    name: str

    def __call__(self, *args, **kwargs) -> Any: ...
