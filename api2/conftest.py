import pytest
import weave

from api2.engine import init_engine


@pytest.fixture
def client():
    client = weave.init_local_client("file::memory:?cache=shared")
    init_engine(client)
