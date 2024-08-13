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
    wc = weave.init_local_client("testableapi.db")
    init_engine(wc)

    word_counts = summarize_curdir_py()
    print(word_counts)
