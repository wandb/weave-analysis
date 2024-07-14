import glob
import pandas as pd
import openai

import weave

from api2.pipeline import BatchPipeline, OpCall, weave_map
from api2.provider import calls, LocalDataframe
from api2.engine import init_engine


@weave.op()
def word_count(code: str):
    word_counts = {}
    for word in code.strip().split():
        if word not in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1
    return dict(reversed(sorted(word_counts.items(), key=lambda x: x[1])))


def summarize_curdir_py():
    files = []
    for f_name in glob.glob("*.py"):
        files.append({"name": f_name, "code": open(f_name).read()})
    map = weave_map(files, word_count, n_trials=2)
    return map.get_result()


if __name__ == "__main__":
    # proj, op = "humaneval6", "ZeroShotOpenaiModel.predict"
    proj, op = "humaneval6", "Evaluation.predict_and_score"
    # proj, op = "weave-hooman1", "QAModel.predict"
    # proj, op = "weave-hooman1", "Evaluation.predict_and_score"
    wc = weave.init(proj)
    init_engine(wc)

    # TODO: hangs with no limit...
    my_calls = calls(wc, op, limit=1000)
    # first_child = my_calls.children().nth(0)
    # child_counts = my_calls.children().count()
    # print(child_counts)
    # first_child = my_calls.children().nth(0)
    # print(first_child)
    # first_child_op_name = first_child.column("op_name")
    # print(first_child_op_name)

    print(my_calls.columns())

    grouped = my_calls.groupby("inputs.model")
    # grouped_df = grouped.to_pandas()
    # print("GROUPED", grouped)
    models = grouped.agg(
        {"output.scores.score_humaneval_test.correct": ["mean", "sem"]}
    )
    group_calls = models.groups()
    print("GROUP CALLS", group_calls)

    # print(models.groups())
    # classes = first_child.input("messages").nth(0).apply(classify)

    # call_child_counts = my_calls.children().counts()
    # vcs = call_child_counts.value_counts()
    # table = Table(vcs)
    # table.add_col("count", vcs.count)
    # table.add_col("first", vcs.group[0])
