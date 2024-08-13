import weave
import openai
from pydantic import field_validator
from weave.trace.vals import TraceTable


class Thread(weave.Object):
    messages: weave.Table

    @field_validator("messages", mode="before")
    def convert_to_table(cls, messages) -> weave.Table:
        rows = messages
        if not isinstance(rows, weave.Table):
            table_ref = getattr(rows, "table_ref", None)
            if isinstance(rows, TraceTable):
                rows = list(rows)
            rows = weave.Table(rows)
            if table_ref:
                rows.table_ref = table_ref
        if len(rows.rows) == 0:
            raise ValueError("Attempted to construct a Dataset with an empty list.")
        for row in rows.rows:
            if not isinstance(row, dict):
                raise ValueError(
                    "Attempted to construct a Dataset with a non-dict object. Found type: "
                    + str(type(row))
                    + " of row: "
                    + short_str(row)
                )
            if len(row) == 0:
                raise ValueError(
                    "Attempted to construct a Dataset row with an empty dict."
                )
        return rows
