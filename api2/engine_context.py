from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api2.engine import Engine

ENGINE = None


def get_engine() -> "Engine":
    if ENGINE is None:
        raise ValueError("Engine not set")
    return ENGINE
