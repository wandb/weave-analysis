# Weave API design

Goal: find the “minimal intuitive factoring” for the API

User stories:

- Experimentation
    - Run different models on the same input
      x Pipeline
    - Run a model for multiple trials, only computing new trials if necessary
      / Pipeline, needs trials
    - Run a model over a batch of inputs, only computing new outputs if necessary.
      x Pipeline
    - Combination of all of above
      / Pipeline, needs trials
- Eval
    - Run a few more trials for an eval, without repeating work, to reduce error bars
      / Pipeline, Needs trials
    - Compare two different models in the same eval
      x Pipeline
    - Change how a particular eval scorer is summarized, without rerunning everything.
      x Pipeline
    - Iterate on the definition of a scorer function, without rerunning the predict functions over and over (reuse prior results)
      x Pipeline
    - Add a scorer that explains failures, then use those explanations to improve prompt.
      - TODO
    - Classify dataset examples and plot scorers based on classes.
      / groupby
    - Plot each example on two score dimensions with error bars
      / groupby
    - Reproduce Eval Compare UI, simply, in code
      - TODO enumerate features
- Prod
    - Given a bunch of production calls for text-extract, classify the input documents (using an op), score the outputs (using an LLM judge), and then plot the group scores
        - Compare two different classification function score distributions
        - Make this computation permanent for production once (”compute this column from now on”)
    - Drift analysis in prod: embed input documents and then get notified when a new example embedding is far from any known ones
    - Cluster production documents, try to explain clusters using LLMs. Make this analysis repeatable.
    - Produce time bucketed class/score plots, efficiently as data streams in over billions of predictions.
- Dataset contruction
    - Fetch all openai child calls from a set of predict calls, and turn into a dataset
    - Periodically find a call and add it to a dataset (remember column mapping)

Goals:

- batch computation
    - cost calculation before computing
- compute locally or on backend
- group by ref-tracking, ie drilldown.
- caching
- dimensions to vary
    - pipeline definition
    - pipeline variants
    - inputs
    - trials

Inspiration:

- pandas
- scitkit-learn


# Summarizing the work

There are two primary new APIs:
- lazy calls() API
- Pipeline/BatchPipeline/ResultTable API
- Execute interface

Examples of using these are in:
- test_api2.py
- programmer_summarize.py
- st_new_api.py ? But I haven't tested this lately.

Missing
- Pipeline must be explicitly called. What if I want to use cache for my op's
  sub-steps?
  - challenge with this: we want to batch-fetch cache results. If we're
    running an op that has sub-steps, many times in parallel, its harder to
    batch (we need to use an async engine and try to aggregate op-calls together,
    or the user needs to use the pipeline object instead of an op).
- Post eval summary/rollups
- Show how UI connects to these APIs to do plots etc.
- Compare two different time-ranges of calls()
- Trials implementation.
- Ref-tracking through to_pandas (see programmer_summarize.py)

NOTES
Bah, thinking about how its hard to explain Pipeline etc.
- Vary/Experiment seems nice. I want to Vary my Data, then call ops on it
- Or Vary my Op, and call it on data.
- I'm building a tree of results.