import pytest
import pandas as pd
from pandas.testing import assert_series_equal

import weave

from api2.cache import batch_get, batch_fill, NOT_COMPUTED


@weave.op()
def classify(doc):
    if ":" in doc or "(" in doc or ")" in doc:
        return "symbols"
    else:
        return "text"


def test_batch_get(client):
    doc0 = "this is a text document"
    doc1 = "this is a doc with symbols :)"
    doc2 = "another text doc"
    docs = pd.DataFrame({"doc": [doc0, doc1, doc2]})

    cache_result = batch_get(docs, classify)
    assert_series_equal(
        cache_result, pd.Series([NOT_COMPUTED, NOT_COMPUTED, NOT_COMPUTED])
    )

    classify(doc0)
    cache_result = batch_get(docs, classify)
    assert_series_equal(cache_result, pd.Series(["text", NOT_COMPUTED, NOT_COMPUTED]))

    delta_iter = batch_fill(docs, classify, cache_result)
    deltas = list(delta_iter)
    assert len(deltas) == 2
    # TODO: check exact deltas (but could be out of order)
    assert_series_equal(cache_result, pd.Series(["text", "symbols", "text"]))

    delta_iter = batch_fill(docs, classify, cache_result)
    deltas = list(delta_iter)
    assert len(deltas) == 0


# def test_batch_fill(client):
#     doc0 = "this is a text document"
#     doc1 = "this is a doc with symbols :)"
#     docs = pd.DataFrame({"doc": [doc0, doc1]})

#     cache_result = batch_get(docs, classify)
#     assert_series_equal(cache_result, pd.Series([NOT_COMPUTED, NOT_COMPUTED]))

#     batch_fill(docs, classify, cache_result)
