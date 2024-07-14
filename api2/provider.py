# TODO: I think this wants to be called "Source" "ColumnSource"?

from typing import Optional, Union, Any
from dataclasses import dataclass
from weave.weave_client import WeaveClient
from weave.trace.refs import CallRef
from weave.trace_server.trace_server_interface import _CallsFilter

import pandas as pd

from api2.proto import Query, QuerySequence, Column, DBOp, ListsQuery
from api2.engine_context import get_engine


@dataclass
class Index:
    level_names: list[str]
    values: list[tuple]


@dataclass(frozen=True)
class CallsQuery(Query):
    from_op: "DBOp"
    _filter: Optional[_CallsFilter] = None
    limit: Optional[int] = None

    # def filter(self, column, value):
    #     return CallsQuery(self.df[self.df[column] == value])

    def groupby(self, col_name):
        return CallsGroupbyQuery(DBOpGroupBy(self, col_name))

    def columns(self):
        return CallsColumnsQuery(DBOpCallsColumns(self))

    def column(self, column_name: str):
        return CallValsQuery(DBOpCallsColumn(self, column_name))

    def children(self) -> "CallsChildrenQuery":
        return CallsChildrenQuery(DBOpCallsChildren(self))
        # return CallsChildrenQuery(self, _CallsFilter())

    def __len__(self):
        engine = get_engine()
        query = ValQuery(DBOpLen(self))
        return engine.execute(query)
        return engine.calls_len(self._filter, limit=self.limit)

    def __str__(self):
        engine = get_engine()
        result = engine.execute(self)
        return str(result)

    def to_pandas(self) -> pd.DataFrame:
        vals = []
        for page in get_engine().calls_iter_pages(self._filter, limit=self.limit):
            vals.extend(page)
        df = pd.json_normalize(vals)
        if df.empty:
            return df
        # df = pd_apply_and_insert(df, "op_name", query.split_obj_ref)
        call_refs = [CallRef(self.entity, self.project, c["id"]).uri() for c in vals]
        df.index = pd.Index(call_refs)
        return df


@dataclass(frozen=True)
class DBOpCallsTableRoot(DBOp):
    entity: str
    project: str


@dataclass(frozen=True)
class DBOpCounts(DBOp):
    input: ListsQuery


@dataclass(frozen=True)
class DBOpNth(DBOp):
    input: ListsQuery
    idx: int


@dataclass(frozen=True)
class DBOpGroupBy(DBOp):
    input: Query
    col_name: str


@dataclass(frozen=True)
class DBOpAgg(DBOp):
    input: Query
    agg: Any

    def __hash__(self):
        return hash((self.input, tuple(list(self.agg))))


@dataclass(frozen=True)
class DBOpLoc(DBOp):
    input: Query
    idx: Any

    def __hash__(self):
        if isinstance(self.idx, list):
            idx = tuple(self.idx)
        else:
            idx = self.idx
        return hash((self.input, idx))


@dataclass(frozen=True)
class DBOpCallsChildren(DBOp):
    input: CallsQuery


@dataclass(frozen=True)
class DBOpCallsColumn(DBOp):
    input: CallsQuery
    col_name: str


@dataclass(frozen=True)
class DBOpCallsColumns(DBOp):
    input: CallsQuery


@dataclass(frozen=True)
class DBOpCallsGroupbyGroups(DBOp):
    input: "CallsGroupbyAggQuery"


@dataclass(frozen=True)
class DBOpLen(DBOp):
    input: Query


@dataclass(frozen=True)
class CallValsQuery:
    from_op: DBOp

    def __str__(self):
        engine = get_engine()
        return str(engine.execute(self))


@dataclass(frozen=True)
class ValQuery:
    from_op: DBOp

    def __str__(self):
        engine = get_engine()
        return str(engine.execute(self))

    def to_pandas(self):
        engine = get_engine()
        return engine.execute(self)


@dataclass(frozen=True)
class CallsChildrenQuery(ListsQuery):
    from_op: DBOp
    _filter: Optional[_CallsFilter] = None

    def groupby(self, col_name):
        return CallsGroupbyQuery(DBOpGroupBy(self, col_name))

    def count(self) -> CallValsQuery:
        return CallValsQuery(DBOpCounts(self))

    def nth(self, idx):
        return CallsQuery(DBOpNth(self, idx))

    def __str__(self):
        engine = get_engine()
        return str(engine.execute(self))

    def to_pandas(self):
        engine = get_engine()
        return engine.execute(self)


@dataclass(frozen=True)
class CallsGroupbyQuery(Query):
    from_op: DBOp

    def agg(self, agg):
        return CallsGroupbyAggQuery(DBOpAgg(self, agg))

    def __str__(self):
        return str(self.to_pandas())

    def to_pandas(self):
        engine = get_engine()
        dfgb = engine.execute(self)
        result = dfgb.apply(lambda x: x)
        result.index = result.index.rename("ref", level=-1)
        return result


@dataclass(frozen=True)
class CallsGroupbyAggQuery(Query):
    from_op: DBOp

    def __str__(self):
        engine = get_engine()
        result = engine.execute(self)
        return str(result)

    def loc(self, idx):
        if not isinstance(idx, list):
            idx = [idx]
        return CallsGroupbyAggQuery(DBOpLoc(self, idx))

    def groups(self):
        return CallsChildrenQuery(DBOpCallsGroupbyGroups(self))

    def to_pandas(self):
        engine = get_engine()
        gb_agg = engine.execute(self)
        return gb_agg.agg


@dataclass(frozen=True)
class CallsQueryColumn(Column):
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
        df.index = pd.Index(call_refs)
        return df[self.column_name]


@dataclass(frozen=True)
class CallsColumnsQuery:
    from_op: DBOp
    column_filter = None  # TODO

    def __str__(self):
        engine = get_engine()
        result = engine.execute(self)
        return str(result)


class LocalDataframeColumn(Column):
    def __init__(self, series):
        self.series = series

    def to_pandas(self) -> pd.Series:
        return self.series


class LocalDataframe(QuerySequence):
    def __init__(self, df):
        self.df = df

    def column(self, column_name: str):
        return LocalDataframeColumn(self.df[column_name])


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
    return CallsQuery(
        DBOpCallsTableRoot(self.entity, self.project), trace_server_filt, limit=limit
    )


SourceType = Union[QuerySequence, pd.DataFrame, list[dict]]


def make_source(val: SourceType) -> QuerySequence:
    if isinstance(val, QuerySequence):
        return val
    elif isinstance(val, pd.DataFrame):
        return LocalDataframe(val)
    elif isinstance(val, list):
        return LocalDataframe(pd.DataFrame(val))
    raise ValueError("Must provide a... TODO")
