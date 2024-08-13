import asyncio
from typing import Callable
import pytest
import tqdm
import pandas as pd
from pandas.testing import assert_series_equal

import weave
from api2 import example_eval
from api2 import example_agent
from api2.cache import NOT_COMPUTED
from api2.engine import init_engine
from api2.provider import calls
from api2.evaluate import eval_lazy
from api2.pipeline import Pipeline

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


def test_read_api_works(client):
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

    # TODO: map API currently disabled. Only pipeline API works for now.
    # Can map.
    # # TODO: This needs to support trials, cost and result streaming!
    # classes = inputs_example.map(classify)
    # assert classes.cost() == {"to_compute": 10}
    # classes_df = classes.to_pandas()
    # # assert classes_df.value_counts().to_dict() == {"text": 4, "symbols": 1}


def test_eval_execute_api_works(client):
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


# This test combines the read and execute APIs
def test_agent_summarize(client):
    agent = example_agent.Agent()
    example_agent.run(agent, example_agent.INITIAL_STATE)
    example_agent.run(agent, example_agent.INITIAL_STATE)

    q = calls(client, "run", limit=10)
    # Each agent run has N agent "step" calls as children.
    q = q.children()
    # Take the last present step, which includes the full conversation history.
    q = q.nth(-1)
    # The "state" parameter is the AgentState which includes the history
    q = q.column("inputs.state")
    # Its a ref to an AgentState object, so expand it.
    q = q.expand_ref()
    # Grab the history attribute, which is the list of messages.
    q = q.column("history")

    df = q.to_pandas()

    # Construct a PipelineResults object.
    p = Pipeline()
    p.add_step(example_agent.summarize_run_rollout)
    bound_p = p.lazy_call({"history": df})
    result_table = bound_p.fetch_existing_results()
    result_table_df = result_table.to_pandas()

    # Prior to execute, there may be un-computed results
    assert len(result_table_df) == 2
    assert result_table_df["summarize_run_rollout_output"].iloc[0] == NOT_COMPUTED
    assert result_table_df["summarize_run_rollout_output"].iloc[1] == NOT_COMPUTED

    # Need to execute 2 summary ops
    assert result_table.execute_cost() == {
        "to_compute": {
            "weave:///none/none/op/summarize_run_rollout:YO6VGCp7thn6OAFnXmpD10sM3D2qVYltLOQpys99lHk": 2
        }
    }

    # Execute the pipeline printing results as they are available.
    for i, delta in enumerate(result_table.execute()):
        pass

    # Got our summaries
    result_table_df = result_table.to_pandas()
    assert len(result_table_df) == 2
    assert (
        result_table_df["summarize_run_rollout_output"].iloc[0]
        == "[{'role': 'system', 'content': 'you are smart'}, {'role': 'user', 'content': 'step 1'}, {'role': 'user', 'content': 'step 2'}]"
    )
    assert (
        result_table_df["summarize_run_rollout_output"].iloc[1]
        == "[{'role': 'system', 'content': 'you are smart'}, {'role': 'user', 'content': 'step 4'}, {'role': 'user', 'content': 'step 5'}]"
    )

    # Nothing left to compute
    assert result_table.execute_cost() == {
        "to_compute": {
            "weave:///none/none/op/summarize_run_rollout:YO6VGCp7thn6OAFnXmpD10sM3D2qVYltLOQpys99lHk": 0
        }
    }
