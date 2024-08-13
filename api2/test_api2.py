import asyncio
from typing import Callable
import pytest
import tqdm
import pandas as pd
from pandas.testing import assert_series_equal

import weave
from api2 import example_eval
from api2.engine import init_engine
from api2.provider import calls
from api2.evaluate import eval_lazy

# from api2.cache import batch_get, batch_fill

# @pytest.fixture
# def profile():
#     with cProfile.Profile() as pr:
#         yield
#     pr.dump_stats('dump.statsVk')


@weave.op()
def classify(doc):
    if ":" in doc or "(" in doc or ")" in doc:
        return "symbols"
    else:
        return "text"


def test_works():
    client = weave.init_local_client("file::memory:?cache=shared")
    init_engine(client)
    eval = weave.Evaluation(dataset=example_eval.dataset, scorers=[example_eval.match])
    res = asyncio.run(eval.evaluate(example_eval.sentiment_simple))
    assert res["match"]["true_count"] == 1
    res = asyncio.run(eval.evaluate(example_eval.sentiment_better))
    assert res["match"]["true_count"] == 4

    c = calls(client, "Evaluation.predict_and_score")

    # Can expand input ref
    inputs_example = c.column("inputs.example").expand_ref()
    inputs_example_df = inputs_example.to_pandas()
    assert len(inputs_example_df) == 10
    assert inputs_example_df.columns.tolist() == ["doc", "sentiment"]

    # Can groupby and agg
    compare_key = "inputs.model"
    x_key = "output.scores.match"
    stats = c.groupby(compare_key).agg({x_key: ["mean", "sem"]})
    stats_df = stats.to_pandas()
    assert stats_df["output.scores.match.mean"].tolist() == [0.8, 0.2]
    assert stats_df["output.scores.match.sem"].tolist() == [0.2, 0.2]

    # Can map.
    # TODO: This needs to support trials, cost and result streaming!
    classes = inputs_example.map(classify)
    assert classes.cost() == {"to_compute": 10}
    classes_df = classes.to_pandas()
    assert classes_df.value_counts().to_dict() == {"text": 4, "symbols": 1}


def test_eval_lazy():
    client = weave.init_local_client("file::memory:?cache=shared")
    init_engine(client)
    eval = weave.Evaluation(dataset=example_eval.dataset, scorers=[example_eval.match])
    t = eval_lazy(eval, [example_eval.sentiment_simple, example_eval.sentiment_better])
    assert t.execute_cost() == {
        "to_compute": {
            example_eval.match.ref.uri(): 10,
            example_eval.sentiment_better.ref.uri(): 5,
            example_eval.sentiment_simple.ref.uri(): 5,
        }
    }

    for delta in t.execute():
        print("DELTA", delta)

    assert t.execute_cost() == {
        "to_compute": {
            example_eval.match.ref.uri(): 0,
            example_eval.sentiment_better.ref.uri(): 0,
            example_eval.sentiment_simple.ref.uri(): 0,
        }
    }

    result_table = t.to_pandas()
    assert_series_equal(
        pd.Series(result_table.columns),
        pd.Series(
            [
                "doc",
                "sentiment",
                "model_op_ref",
                "model_output",
                "match_op_ref",
                "match_output",
            ]
        ),
    )

    assert_series_equal(
        result_table["model_op_ref"],
        pd.Series(
            [example_eval.sentiment_simple.ref.uri()] * 5
            + [example_eval.sentiment_better.ref.uri()] * 5,
            name="model_op_ref",
        ),
    )
    assert_series_equal(
        result_table["model_output"],
        pd.Series(
            [
                "positive",
                "negative",
                "positive",
                "neutral",
                "neutral",
                "positive",
                "negative",
                "neutral",
                "positive",
                "negative",
            ],
            name="model_output",
        ),
    )

    assert_series_equal(
        result_table["match_op_ref"],
        pd.Series(
            [example_eval.match.ref.uri()] * 10,
            name="match_op_ref",
        ),
    )
    assert_series_equal(
        result_table["match_output"],
        pd.Series(
            [
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
                True,
                True,
            ],
            name="match_output",
            dtype=object,
        ),
    )
