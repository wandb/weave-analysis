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
from api2.pipeline import BatchPipeline, Pipeline, ResultTable
from api2.cache import batch_get, batch_fill

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

    # questions to answer from here
    # 1 can I define an Eval in terms of weave_map BatchPipeline etc?
    #   - run it, then add more trials, ensure cache hits, ensure metrics go down
    # 2 can I classify the inputs, then group by the classes, and produce a plot
    # 3 can I incrementally build a UI table of expanded example, classify, etc.

    # Would be so cool if I could do a streaming version! That can also stream
    #   back computational results.

    # Big items left:
    #   - define eval
    #   - cost calculation
    #   - result streaming
    #   - table view
    #   - summary metrics
    #   - write streaming

    # Insight
    #   - only really care about streaming individual op results, and those
    #     are traced already, so stream the trace?
    #   - well, no, its nice to stream the summary metrics for an eval
    #   - so eval should return the table and the summary metrics
    #     and both should be streamed! Table is compute table, summary metrics
    #     are derived.
    #   - you can do get_result and get the final result
    #     or you can do for x in... and iterate through the deltas.


# Goal: write tests that build up to the full API


def test_single_call(client):
    assert classify("some nice text") == "text"

    x = lazy_call(classify, doc="some nice text")

    x = Pipeline()
    x.add_step(classify)
    assert x.free_inputs() == ["doc"]

    called = x.lazy_call(doc="some nice text")  # could be called bind?
    assert called.cost() == {"to_compute": 1}
    for delta in called.execute():
        pass
    assert called.get_result() == "text"


# OK I'm trying to write higher level APIs here.... this is what I need.
# and need to make sure I can do an awesome live updating / stremaing UI
# in streamlit based on this.


def test_compare_two_ops(client):
    compared = weave.compare_ops(
        example_eval.sentiment_simple, example_eval.sentiment_better
    )
    called = compared.batch_call(weave.dataset(...))
    for delta in called.execute():
        pass
    print(called.get_result())


def test_classify_prod(client):
    prod_calls = weave.calls("some-prod-op")
    df = prod_calls.group_by(prod_calls.map(classify))["summary.tokens"].mean()
    plot = weave.Plot(
        inputs=prod_calls,
        x=classify,
        # x={"classifier1": classify, "classifier2": classify.versions[-2]},
        y=prod_calls.summary["tokens"],
    )
    for delta in plot.execute():
        pass
    print(plot.get_result())


# I played with a Vary class in a notebook for varying classify above, in Pandas.
# We need to do melt after to normalize, which is expected. But if there were
#   downstream calls that were varied, this wouldn't exactly work as is.
# So what ComputeTable is v DataFrame
#   add computed columns
#   they can depend on others
#   you can vary computations
#   you can have trials
#   you can derive stuff immediately
#   you can see how much computation is required
#   you can say to run the desired computations

# How would it work for classify prod?
#
# f = weave.Frame(weave.calls(...))
# f.add_op(classify)
# frame_cost = f.cost()
# g = f.groupby("classify")
# p = g.agg({"summary.tokens": "mean"})
# plot_cost = p.cost()
# plot_data = p.to_pandas()  # can get an immediate result from cached values.
# p.execute()  # execute only stuff needed to complete p
# f.execute()  # execute everything to complete f

# OK without Frame?
# f = weave.Frame(weave.calls(...))

# # Need to define what this does. Show all results by default? Or just the first
# # or last? What if I set a limit
# classes = f.apply(classify, trials=N)
# # or
# classes = f.apply(Vary(classify, classify.versions[-2]), trials=N)
# g = f.groupby(classes)
# p = g.agg({"summary.tokens": "mean"})
# plot_cost = p.cost()
# plot_data = p.to_pandas()  # can get an immediate result from cached values.
# p.execute()  # execute only stuff needed to complete p
# f.execute()  # execute everything to complete f


# # How would it work for eval

# eval = ...
# f = weave.Frame()
# f.add_slot("predict")
# for scorer in eval.scorers:
#     f.add_op(scorer)
# f.add_input(dataset)
# # f is now an eval

# # now to run it
# f.add_op_to_slot("predict", predict1)
# f.add_op_to_slot("predict", predict2)

# summary = f.summary()  # produces a compared summary, using any avail cache

# f.fill(trials=3)

# # What if we had a more Task oriented API

# task = Task("predict")

# The Eval Builder UI is interesting:
#   show all "models" that are avaialble. But some haven't been fully eval'd
#   for the current definition. But we can find ones to recommend evaling for
#   that definition.
# I need the mechanism of a "partial" result. One that is not fully computed but
#   that can be.


# OK so some learnings:
# - I think I want the definition of apply from above, where I can apply a
#   function, or multiple functions, or trials
# - I want the definition of a "partial" result, one that is not fully computed
# - I think I need a Table thing, but maybe the most illustrative move would
#   be to try to implement the Eval framework on top of the above, now that
#   I have these definitions.
# - And then try to implement the "classify prod" example on top of that.


# def eval_models(eval: Eval, models: list[Model]):
#     f = weave.Frame(eval.dataset)
#     f.add_column(Vary(models), "model")
#     for scorer in eval.scorers:
#         f.add_column(scorer)
#     for delta in f.execute():
#         print(f.summary())
#     # THIS IS NICE!
#     # Can f.summary be implemented as this partial thing?


# # NEW
# # Do I need to be able to build this Frame, or can I just call stuff?
# # I need a Frame so I have a single thing I can use to call cost/execute on


# def eval_models(eval: Eval, models: list[Model]):
#     d = eval.dataset
#     model_outputs = d.apply(Vary(models, "model_output"))
#     scores = []
#     for scorer in eval.scorers:
#         scores.append(model_outputs.apply(scorer))
#     t = Table(model_outputs, scores)
#     t.agg()  # use default summarizers


# class Eval:
#     def eval_models(self, models):
#         pass

#     def summary(self):
#         pass


# def test_eval():
#     eval = weave.Evaluation(dataset=example_eval.dataset, scorers=[example_eval.match])
#     eval.set_models([example_eval.sentiment_simple, example_eval.sentiment_better])
#     eval.set_trials(3)
#     eval.cost()  # -> {"to_compute": ...}
#     eval.summary()  # -> Incomplete(Summary)
#     for delta in eval.execute():
#         ...


# Again I find myself circling. What am I circling around?
# OK well, I come to the same conclusion. I have all the pieces and requirements
#   down. Now its a matter of finding nice simple APIs, and that's why I'm circling.
# To do that, I can go from both ends.
# From user end: design the eval API as I am doing in test_eval above, and make
#   all of the streamlit apps work on this API
# From the engine end: I need to make apply work with Vary and trials. Output of
#   that is a Dataframe.
#   - and note that operations that output a Series-like thing can name that
#     thing after the function (so we can get named columns without having the
#     the Table API [though we will still want the Table API built on top of
#     of that].


# From plane:
# Above is a good note.
# Two options for Evaluation.evalute_lazy (new API)
#   - it can return Node like old Weave, but with new features like cost
#   - or it can return a ComputeTable.
# And this is why I need to do the design from both sides. Just need to start
#   implementing Eval and playing with it to see what happens.


def eval_lazy(
    eval: weave.Evaluation, models: list[Callable], n_trials: int = 1
) -> ResultTable:
    # dataset = pd.DataFrame(list(eval.dataset.rows))
    # TODO: Repeat data with weave_trial param column
    pipeline = Pipeline()

    # First, add our model calls. Using the same step_id means we will
    # call each model in that position
    for model in models:
        pipeline.add_step(model, step_id="model")

    # Next, call add each scorer call. Scorers typically don't depend on eachothers'
    # results, but we don't need to manually specify that here. (however, if they
    # do depend on each other's results, this will just work!)
    for scorer in eval.scorers:
        pipeline.add_step(scorer)

    # Here we bind out pipeline to our dataset, for the number of trials we want.
    # An alternative would be eval.dataset.lazy_apply(pipeline, n_trials)
    bound_pipeline = pipeline.lazy_call(eval.dataset, n_trials=n_trials)

    # Next we fetch compute our results table, which may have NOT_COMPUTED values
    result_table = bound_pipeline.fetch_existing_results()

    return result_table


def test_eval_lazy():
    client = weave.init_local_client("file::memory:?cache=shared")
    init_engine(client)
    eval = weave.Evaluation(dataset=example_eval.dataset, scorers=[example_eval.match])
    t = eval_lazy(eval, [example_eval.sentiment_simple, example_eval.sentiment_better])
    assert t.remaining_cost() == {
        "to_compute": {
            example_eval.match.ref.uri(): 10,
            example_eval.sentiment_better.ref.uri(): 5,
            example_eval.sentiment_simple.ref.uri(): 5,
        }
    }

    for delta in t.fill():
        print("DELTA", delta)

    assert t.remaining_cost() == {
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


def format_progress_line(cost, summary):
    # TODO: this should be a nice progress line!
    return f"{cost} {summary}"


# This is how things work if you just run in the terminal.
def eval_run(
    eval: weave.Evaluation, models: list[Callable], n_trials: int = 1
) -> ResultTable:
    t = eval_lazy(eval, models, n_trials)
    # t.fill() begins executing all the necessary operations for computing results.
    progress_bar = tqdm.tqdm(t.fill())
    # We can iterate through t.fill()'s returned iterator to see the deltas (each
    # op's result)
    for delta in progress_bar:
        # t.summary() uses default summarizers (which we need to generalize), unless
        # those are overridden (APIs not yet defined). Note, this can return a PartialValue
        # object (also not yet designed), since there are at least some NOT_COMPUTED
        # results still.
        t_summary = t.summary()
        # t.remaining_cost() tells us what ops still need to be computed (counts)
        # and can eventually include estimated timing, $, etc.
        t_cost = t.remaining_cost()
        # Show the user the cost and summary as results are computed.
        progress_bar.set_description(format_progress_line(t_cost, t_summary))

    return t


# Awesome.
# Now design what it looks like when streaming into a UI (streamlit / js)
# Here we want a nice simple result table as the results come in.
#   This should be done using some sort of View composition thing... Question,
#      is .summary() above the same?
# This is nice, major progress!
# My main note on above is its harder to explain how the whole thing is turned
# into a Table, than if the API itself was the ComputeTable API I designed before.
# Like we want to think of adding a "model" column to a virtual table, logically.
#   Well, I can at least write a really nice doc on that.
#   "Think of a Pipeline as the definition of columns in a table..."
#   (with diagrams)
#   "A ComputeTable is composed of a Pipeline and a DataProvider".
#
# OK and I still need to figure out what happens when you want to do stuff like
# groupby on the backend on top of this.
# And I'm wondering if there is a way to manually add Eval results into a Table
# like this, without needing to use our execution framework.
# And I need to add a limit= parameter! Is that part of the "view" or part of
#   the underlying compute table?
#
# But jump to implementing this for real next, and then make the interactive
# Streamlit thing. That's will resolve the next set of big open questions, because
#   we want to use the same Data Model for editing (like I want to change the code
#   in an op one column, and run again and see what happens, for example).
