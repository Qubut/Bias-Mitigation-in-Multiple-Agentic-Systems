# `MASStateMachine` API

The `MASStateMachine` is the deterministic lifecycle engine for one MAS run.

## Lifecycle Overview

The machine executes three states in sequence:

1. `genesis` (initial): each agent generates a turn-0 answer.
2. `interaction`: agents update answers round-by-round from peer responses.
3. `completed` (final): run terminates and full prediction history is returned.

## Transitions

- `to_interaction`: `genesis -> interaction`
- `continue_interaction`: `interaction -> interaction`
- `finish`: `interaction -> completed`

Termination occurs when `current_round > config.rounds`.

## API Documentation

```{eval-rst}
.. automodule:: bias_mitigation.mas.mas_statemachine
   :members:
   :exclude-members: genesis, interaction, completed, to_interaction, continue_interaction, finish, states, states_map, initial_state, final_states
   :show-inheritance:
```
