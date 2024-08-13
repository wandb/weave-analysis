import weave
import openai
import textwrap
from api2.engine import init_engine
from api2.provider import calls
from api2.pipeline import Pipeline


@weave.op()
def summarize_run_rollout(history):
    # Data fetching logic
    history_s = str(history)

    # LLM logic
    history_truncated_s = ""
    if len(history_s) > 150 * 1024:
        history_s = history_s[: 150 * 1024] + "..."
        history_truncated_s = "HISTORY TRUNCATED\n"

    prompt = textwrap.dedent(
        f"""
    You will be provided with a conversation between a user and an LLM agent that has access to the local filesystem.

    <conversation>
    {history_s}
    </conversation>
    {history_truncated_s}

    The conversation proceeds by the user providing input (a "user step") followed by N "assistant steps" where the assistant does work.

    The first user input in a conversation defines a Task, which has a Goal.

    Each user step we encounter may do one of the following actions:
    - Update Goal for current Task
    - Replace Goal starting new Task
    - Provide Guidance for current Task starting SubTask

    The Task and SubTask cannot change unless the user intervenes!

    Please provide summary:
    - User Task: <task definition summary>
      - User SubTask: <subtask definition summary>
        - Assistant: <assistant step summary>
        - Assistant: <assistant step summary>
        - ...
      - ...
      - Success: [true|false|partial]
    - ...
    """
    )
    response = openai.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


client = weave.init("shawn/programmer-weave-analytics-pipeline")
init_engine(client)

# Fetch the data we want to operate on. In this case, we want to summarize
# the last step of each agent run.
q = calls(client, "run", limit=10)
# Each agent run has N agent "step" calls as children.
q = q.children()
# Take the last present step, which includes the full conversation history.
q = q.nth(-1)
# The "state" parameter is the AgentState which includes the history
q = q.column("inputs.state")
# Its a ref to an AgentState object, so expand it.
q = q.expand_ref()
# Grab the history attribute, which is the list of messages.
q = q.column("history")

df = q.to_pandas()

# Construct a PipelineResults object.
p = Pipeline()
p.add_step(summarize_run_rollout)
bound_p = p.lazy_call({"history": df})
result_table = bound_p.fetch_existing_results()

# Prior to execute, there my be un-computed results
print()
print("RESULT TABLE PRE-EXECUTE")
print(result_table.to_pandas())

print()
print("RESULT TABLE COST")
print(result_table.execute_cost())

# Execute the pipeline printing results as they are available.
print()
for i, delta in enumerate(result_table.execute()):
    print("GOT RESULT", i, delta)

print()
print("RESULT TABLE POST-EXECUTE")
print(result_table.to_pandas())


# TODO: this does not track refs correctly, we lost the history ref!
