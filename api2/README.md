Weave API Design
================

Goals:
- Basic Calls Read API
  - We need to ship a public, python-based calls Read API, so users can read a bunch of calls back and do stuff.
  - It should have a to_pandas() method
  - Lift up parameters on existing .calls() API
  - Batch methods for fetching input columns, children, etc.
  - Refs read batch method for reading a batch of refs.
- Ability to easily add more trials to an eval
- Ability to launch evals from UI
- And build a playground, etc.
- And LLM-driven UI.
- Simple APIs for batching/mapping and concurrency control
- Simple APIs for comparing two different functions/models
- Ability to log evals step by step
- Ability to use our eval API without running our eval framework
- Ability to add derived values to calls and use them in analysis

Again, a similar conclusion. Eval is already a great container for what this is.
- I need to just make that API nicer, by using this new stuff (ComputeTable)
- Still, Eval should be composed of these other things like Pipeline.
  - The nice thing about ComputeTable is it tells you how much work there is to do.
- But I should be able to define what we need to do for the .calls() read basic
  interface now, without doing above?

But I could design this "from the API", ie just make a bunch of examples for all the use cases,
  without actually doing the full implementations.

Executable:
- A computational system that can tell you how long its going to take to execute is really useful. And with the types of computations we do with LLMs, we can actually make these estimates.
- The Executable API has two methods: cost() and execute()
- First we define a computation. How? Instead of calling ops directly, we use Pipeline, Batch, etc, to build a computation.

The speed at which you can experiment dramatically changes how much progress you can make. Weave helps in a few ways:
- it can accurately tell you how long a computation will take
- it can use parallelism to make things go fast
- it can cache results, so that you can incrementally compute stuff
- it can use approximate caching, to reuse stuff even more
- it can tell you much repetition you need to reduce error bars
- it can minimize datasets to the useful set?

Most important top-level goal:
- Unlock: "What is the Call Read API"
- Arg, just make the stuff work, stop thinking about it!
  - I can do that as soon as I unlock the Calls read API?

I definitely want to get to an answer on "what is the Calls Read API".

Goals:
- memoization
- "run record": the ability to save a record of which calls were used for an analysis
- ability to navigate trace trees in batch
- automatic parallelization with easy controls for rate limits
- multiple trials
- experimentation
- remote and local execution
- API designed to allow us to build local features and then make them more efficient by moving to the backend
- contains everything needed to build data applications on top of Weave
- consumable by LLMs
- add column to call to classify
- add embedding to call input
- semantic caching?

To discuss
- pandas-like interface (pandas data model)

model_a = partial(model, ...)
model_b = partial(model, ...)
comparer = compare(model_a, model_b)
comparer(x)

trialer = trials(model_a, 5)
trialer(hello)

scorer = pipeline(model_a, [score1, score2, score2])
scorer('bla')

Our goal is to build a function.
So first:
  define our task
    (this is a type definition)
  now make a simple function
  add scorers, ability to check results
    possibly requiring ground truth
  try it over a few examples
  try new versions of it
