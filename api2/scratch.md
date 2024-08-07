what would it be like to separate batch compute from query API?

well, you still might want to use batch compute in the query API?

and you still want to have cost analysis in a query API.

query API is superset of batch compute API

well let's just try implementing cost to see how it goes

I want to write tests for everything in Weave API Design. Starting with the simplest
case where we call one op and we want to use the cache for the result.

Let's break down the objects we have.
Pipeline:
  a sequence of connected ops
  open: has unbound input (variable)
  closed: has no variables

Well, I need a Graph definition like Weave1. But can I do it nicer, like
  preserving "let"?

If I can make:
    - streaming
    - cache
    - pipeline
    - comparison
    - partial (Object)
    - batch
    - trials
    - pivot 
    - trials pushdown
  all work together in a simple API, I have something nice.

what if i have a compare pipeline with two compared steps, what is the output table?
  - predict: (2)
    - score1: (2)
    - score2: (1)
  - we don't need to recompute score2 for each score1. We have a tree of results
    here. But we're trying to flatten

  pA, sA1: {p, s1, s2}

  Its ok if we have dupe results? I think so.

  OK the design area is:
  - i have two ways to define a pipeline. BatchPipeline for ComputeTable like
    batch/map
  - and the DBOp API for Weave1 like lazyness.
  - what I want is Weave1 Tables with compute and caching, and all the other
    features listed above.

Weave is an API for Experimental programming

Lazy call and so on are too low-level/generic. I should work on the high level API