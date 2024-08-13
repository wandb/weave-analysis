import glob
import pandas as pd
import openai

import weave

from api2.execute_api import BatchPipeline, OpCall, weave_map
from api2.query_api import calls, LocalDataframe
from api2.engine import init_engine


@weave.op()
def summarize_purpose(code: str):
    prompt = f"Summarize the purpose of the following code, in 5 words or less.\n\n{code}\n\nPurpose:"
    response = openai.chat.completions.create(
        model="gpt-4o", messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content


@weave.op()
def classify(code: str):
    prompt = f"Classify the following code. Return a single word.\n\n{code}\n\nClass:"
    response = openai.chat.completions.create(
        model="gpt-4o", messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content


@weave.op()
def strlen(s: str):
    return len(s)


def summarize_curdir_py():
    files = []
    for f_name in glob.glob("*.py"):
        files.append({"name": f_name, "code": open(f_name).read()})
    map = weave_map(
        LocalDataframe(pd.DataFrame(files)), OpCall(summarize_purpose), n_trials=1
    )
    print(map.cost())
    for delta in map.execute():
        pass
    return map.result


if __name__ == "__main__":
    wc = weave.init_local_client("testableapi.db")
    init_engine(wc)

    print("SUMMARIZE RESULT", summarize_curdir_py())

    # # Read
    my_calls = calls(wc, "summarize_purpose", limit=100)
    print("# Calls", len(my_calls))
    print("Columns", my_calls.columns())
    # my_calls.append_column(classify, my_calls.column("inputs.code"))
    # my_calls.append_column(strlen, my_calls.column("output"))
    # my_calls.cost()
    # grouped = my_calls.groupby("classify")
    # grouped.add_column("mean", grouped.column('strlen')
    # print('grouped cost', grouped.cost())
    # print(grouped)

    # input_code_class = weave_map(my_calls, classify, {"code": "inputs.code"})
    # print("Classify cost", input_code_class.cost())
    # for result in input_code_class.execute():
    #     pass

    # output_len = weave_map(my_calls, strlen, {"s": "output"})
    # print("Strlen cost", input_code_class.cost())
    # for result in output_len.execute():
    #     pass

    # both = pd.concat([input_code_class.result, output_len.result], axis=1)
    # print(both.groupby("classify").mean())

    # TODO: get this grouping down into API
    # show how to fetch all calls in a group. (streamlit example)
    # show fetching up child calls
    # show working
    # Make work into pivot table
    # index (input_ref, trial), columns: one level for each branch

    p = BatchPipeline(my_calls)
    p.add_step(classify, column_mapping={"code": "inputs.code"})
    p.add_step(strlen, column_mapping={"s": "output"})
    print(p.cost())
    print(p.get_result().groupby("classify").mean())

    my_calls = calls(wc, "summarize_purpose", limit=100)
    p = BatchPipeline(my_calls)
    p.add_step(classify, my_calls.column("inputs.code"))

    # TODO:
    # Goal of this whole exercise: build up something that can
    #    replace predict_and_score fetch in my st_app_eval, with something
    #    that gives us control of fetching the eval. And allows stuff
    #    like grouping by classes.
    #   - make compare_step work
    #   - make n_trials work
    #   - better argument passing (instead of column_mapping?)
    #   - non-batch version
    #   - make work on raw data?
    #   - better error handling
    #   - I need whole Call information available, not just output
    #     (for example, for building a nice Eval table with feedback)
    #   - Make a materialized version so that we can actually store a record
    #     of all the calls we've fetched.

    # res = p.execute()
    # print(res)

    # This is how BatchPipeline would work for an Eval

    # eval = eval_picker()
    # ds = eval.dataset
    # models = model_picker()
    # p = BatchPipeline(ds)
    # model_step = p.add_compare_step(step_name="model", n_trials=eval.n_trials)
    # for model in models:
    #     model_step.add_op(model.op)
    # for score_fn in eval.score_fns:
    #     p.add_step(score_fn.op, {})

    # print(t.cost())
    # for result_delta in t.execute():
    #     print(result_delta)
    # df = t.result.dataframe()
