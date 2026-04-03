# `AgentStateMachine` API

The agent lifecycle state machine controls per-agent phase transitions:

- `genesis` (initial)
- `interaction`
- `completed` (final)

Transition behavior:

- genesis step with no peer answers: remains `genesis`
- first step with peer answers: `genesis -> interaction`
- subsequent interaction steps: `interaction -> interaction`
- completion signal: `genesis|interaction -> completed`

```{eval-rst}
.. automodule:: bias_mitigation.mas.agent_statemachine
   :members:
   :undoc-members:
   :show-inheritance:
```
