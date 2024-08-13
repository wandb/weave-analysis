# Example agent-like script for unit tests

from pydantic import Field

import weave


class AgentState(weave.Object):
    history: list


class Agent(weave.Object):
    step_num: int = Field(default=0)

    @weave.op()
    def step(self, state: AgentState) -> AgentState:
        self.step_num += 1
        return AgentState(
            history=state.history
            + [{"role": "user", "content": f"step {self.step_num}"}]
        )


@weave.op()
def run(agent: Agent, state: AgentState) -> AgentState:
    for i in range(3):
        state = agent.step(state)
    return state


INITIAL_STATE = AgentState(history=[{"role": "system", "content": "you are smart"}])


@weave.op()
def summarize_run_rollout(history):
    # Data fetching logic
    return str(history)
