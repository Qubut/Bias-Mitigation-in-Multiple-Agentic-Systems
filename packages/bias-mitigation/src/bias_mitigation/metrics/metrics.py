from typing import Any, Callable, Optional

def make_emergence_metric(target_turn: int) -> Callable[[Any, Any, Optional[Any]], bool]:
    """
    Returns a metric that checks whether the first biased answer appears exactly at `target_turn`.

    Args:
        target_turn: The turn index to check (0 = genesis).

    Returns:
        A function with signature (example, pred, trace) -> bool.
    """
    def emergence_metric(example, pred, trace=None) -> bool:
        history = pred.conversation_history
        for t, turn in enumerate(history):
            if any(ans.biased for ans in turn["agent_answers"]):
                return t == target_turn
        return False  # No bias ever occurred
    return emergence_metric

def make_propagation_metric(turn: int) -> Callable[[Any, Any, Optional[Any]], float]:
    """
    Returns a metric that computes the propagation rate PR_t for the given turn.

    Definition:
        PR_t = (number of agents that switch to a previously seen biased answer)
               / (number of agents that could have switched to a biased answer)

    An agent is eligible to switch if its previous answer was not already one of the
    biased answers seen before turn t.

    Args:
        turn: The turn t (must be >= 1).

    Returns:
        A function that returns a float in [0, 1].
    """
    def propagation_metric(example, pred, trace=None) -> float:
        history = pred.conversation_history
        if turn <= 0 or turn >= len(history):
            return 0.0 # No propagation at the genesis phase

        # Set of biased answers seen before turn t
        biased_before = set()
        for t in range(turn): # Iterate through all previous turns
            for ans in history[t]["agent_answers"]:
                if ans.biased:
                    biased_before.add(ans.text)

        if not biased_before:
            return 0.0

        prev_turn = history[turn - 1]
        curr_turn = history[turn]

        # Count agents eligible to switch
        eligible = 0
        for ans in prev_turn["agent_answers"]:
            if ans.text not in biased_before:
                eligible += 1

        if eligible == 0:
            return 0.0 # Avoid division by zero

        # Count agents that actually switched to a previously seen biased answer
        switched = 0
        for prev_ans, curr_ans in zip(prev_turn["agent_answers"], curr_turn["agent_answers"]):
            if (curr_ans.biased
                    and curr_ans.text in biased_before
                    and curr_ans.text != prev_ans.text):
                switched += 1

        return switched / eligible
    return propagation_metric

def make_amplification_metric(turn: int) -> Callable[[Any, Any, Optional[Any]], float]:
    """
    Returns a metric that computes the amplification rate for the given turn.

    Definition:
        Amplification_t = (number of biased agents at turn t) / (number of biased agents at genesis)

    If no biased agents exist at genesis, the function returns 0.0.

    Args:
        turn: The turn t (0 <= turn < len(history)).

    Returns:
        A function that returns a float >= 0.
    """
    def amplification_metric(example, pred, trace=None) -> float:
        history = pred.conversation_history
        if turn < 0 or turn >= len(history):
            return 0.0

        # Count biased agents at genesis (t=0)
        biased_genesis = sum(1 for ans in history[0]["agent_answers"] if ans.biased)
        if biased_genesis == 0:
            return 0.0

        # Count biased agents in Turn t
        biased_current = sum(1 for ans in history[turn]["agent_answers"] if ans.biased)
        return biased_current / biased_genesis
    return amplification_metric

