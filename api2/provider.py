from typing import Optional, Union, Any
from dataclasses import dataclass
from weave.weave_client import WeaveClient
from weave.trace.refs import CallRef
from weave.trace_server.trace_server_interface import _CallsFilter

import pandas as pd

from api2.proto import QuerySequence, Column
from api2.engine_context import get_engine


@dataclass
class Index:
    level_names: list[str]
    values: list[tuple]


@dataclass
class CallsQuery(QuerySequence):
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

    def column(self, column_name: str):
        return CallsQueryColumn(self, column_name)

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
        df.index = pd.Index(call_refs)
        return df


class CallsQueryGroupby:
    calls_query: "CallsQuery"
    by: "CallsQueryColumn"

    def column(self):
        pass


@dataclass
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


class LocalColumn(Column):
    def __init__(self, series):
        self.series = series

    def to_pandas(self) -> pd.Series:
        return self.series


class LocalQueryable(QuerySequence):
    def __init__(self, df):
        self.df = df

    def column(self, column_name: str):
        return LocalColumn(self.df[column_name])


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


SourceType = Union[QuerySequence, pd.DataFrame, list[dict]]


def make_source(val: SourceType) -> QuerySequence:
    if isinstance(val, QuerySequence):
        return val
    elif isinstance(val, pd.DataFrame):
        return LocalQueryable(val)
    elif isinstance(val, list):
        return LocalQueryable(pd.DataFrame(val))
    raise ValueError("Must provide a... TODO")
