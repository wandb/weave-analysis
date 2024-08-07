import asyncio
import weave


@weave.op
def sentiment_simple(doc):
    if "yes" in doc or "😀" in doc:
        return "positive"
    if "no" in doc or "😢" in doc:
        return "negative"
    return "neutral"


@weave.op
def sentiment_better(doc):
    score = 0
    for pos_word in ["yes", "better"]:
        if pos_word in doc:
            score += 1
    for neg_word in ["no", ":("]:
        if neg_word in doc:
            score -= 1
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"


@weave.op
def match(model_output, sentiment):
    return model_output == sentiment


dataset = weave.Dataset(
    rows=[
        {"doc": "yes I'll join you", "sentiment": "positive"},
        {"doc": "no, he did pass!", "sentiment": "positive"},
        {"doc": "yes, no, I'm not sure", "sentiment": "neutral"},
        {"doc": "It couldn't have gone better", "sentiment": "positive"},
        {"doc": ":( :( :(", "sentiment": "negative"},
    ]
)


if __name__ == "__main__":
    weave.init_local_client("example_eval.db")
    eval = weave.Evaluation(dataset=dataset, scorers=[match])
    res = asyncio.run(eval.evaluate(sentiment_simple))
    print("sentiment_simple res: ", res)
    res = asyncio.run(eval.evaluate(sentiment_better))
    print("sentiment_better res: ", res)
