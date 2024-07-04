from typing import Any, Iterator, Protocol

import pandas as pd


class Executable(Protocol):
    def cost(self) -> dict: ...

    def execute(self) -> Iterator[dict]: ...

    def get_result(self) -> Any: ...


# Really this is like a Pandas series
# it has a name (which could be a tuple?) and an index.
# So spec this out more.
class Column(Protocol):
    def to_pandas(self) -> pd.Series: ...


class QuerySequence(Protocol):
    def column(self, column_name: str) -> Column: ...


class OpType(Protocol):
    name: str

    def __call__(self, *args, **kwargs) -> Any: ...
