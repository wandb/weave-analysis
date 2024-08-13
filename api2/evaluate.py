import tqdm
from typing import Callable
import weave

from api2.execute_api import Pipeline, PipelineResults


def eval_lazy(
    eval: weave.Evaluation, models: list[Callable], n_trials: int = 1
) -> PipelineResults:
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

    # Here we bind our pipeline to our dataset, for the number of trials we want.
    # An alternative would be eval.dataset.lazy_apply(pipeline, n_trials)
    bound_pipeline = pipeline.lazy_call(eval.dataset, n_trials=n_trials)

    # Next we fetch compute our results table, which may have NOT_COMPUTED values
    result_table = bound_pipeline.fetch_existing_results()

    return result_table


def format_progress_line(cost, summary):
    # TODO: this should be a nice progress line!
    return f"{cost} {summary}"


# TODO: this is unused and untest. We don't implement t.summary yet.
def eval_run(
    eval: weave.Evaluation, models: list[Callable], n_trials: int = 1
) -> PipelineResults:
    t = eval_lazy(eval, models, n_trials)
    # t.fill() begins executing all the necessary operations for computing results.
    progress_bar = tqdm.tqdm(t.execute())
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
        t_cost = t.execute_cost()
        # Show the user the cost and summary as results are computed.
        progress_bar.set_description(format_progress_line(t_cost, t_summary))

    return t
