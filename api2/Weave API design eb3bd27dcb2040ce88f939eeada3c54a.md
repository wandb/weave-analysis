# Weave API design

Goal: find the “minimal intuitive factoring” for the API

User stories:

- Experimentation
    - Run different models on the same input
    - Run a model for multiple trials, only computing new trials if necessary
    - Run a model over a batch of inputs, only computing new outputs if necessary.
    - Combination of all of above
- Eval
    - Run a few more trials for an eval, without repeating work, to reduce error bars
    - Compare two different models in the same eval
    - Change how a particular eval scorer is summarized, without rerunning everything.
    - Iterate on the definition of a scorer function, without rerunning the predict functions over and over (reuse prior results)
    - Add a scorer that explains failures, then use those explanations to improve prompt.
    - Classify dataset examples and plot scorers based on classes.
    - Plot each example on two score dimensions with error bars
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

TODO:

- cache sub-steps (like retrieval v. generation)
- fetch child call columns