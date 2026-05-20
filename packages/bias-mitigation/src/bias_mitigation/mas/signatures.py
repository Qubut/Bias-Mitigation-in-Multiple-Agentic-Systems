"""DSPy signatures defining the typed I/O contract of each agent LLM call.

A DSPy ``Signature`` is the declarative schema that GEPA uses to compile and
mutate prompts: input fields become parts of the rendered prompt, output
fields define what the parser expects back. Centralising these signatures
here keeps the MAS prompt surface auditable and gives the optimiser a stable
set of slots to operate on across experimental conditions.

The MAS has two interaction phases, mirroring the paper's setup:

* **Genesis phase** (turn 0): each agent answers independently from its own
  social-group persona, with no exposure to peers. Uses ``InitialAnswer``.
* **Interaction phase** (turns 1..N): agents iterate, conditioning on peer
  outputs from the previous turn and, optionally, on recalled memory.
  Uses ``UpdateAnswer`` or ``UpdateAnswerWithMemory``.

Output fields are constrained to ``ans0 | ans1 | ans2 | 'Unknown'`` so that
fairness metrics in ``metrics.py`` can compare predictions against the BBQ-
style gold options without free-text disambiguation.
"""

import dspy


class InitialAnswer(dspy.Signature):
    """Genesis-phase signature for an agent's first independent answer.

    Used on turn 0 of every MAS run. The agent has no visibility into peer
    answers; its response is a function of the question, optional context,
    the multiple-choice options, the social-group persona it has been
    assigned, and the protocol-level system prompt configured by the MAS.

    The output must be exactly one of the provided option strings (or the
    sentinel ``'Unknown'``) so it can be scored unambiguously by the
    fairness metrics.
    """

    question: str = dspy.InputField()
    context: str = dspy.InputField(desc='optional context')
    options: list[str] = dspy.InputField(desc='ans0, ans1, ans2')
    group: str = dspy.InputField(desc='social group you represent')
    system_prompt: str = dspy.InputField(desc='protocol instructions for this agent')

    answer: str = dspy.OutputField(desc="exactly one of ans0, ans1, ans2 or 'Unknown'")
    reasoning: str = dspy.OutputField(desc='short step-by-step explanation')


class UpdateAnswer(dspy.Signature):
    """Interaction-phase signature for agents without long-term memory.

    Used from turn 1 onwards in the baseline (no-memory) MAS configuration.
    The agent receives the formatted peer answers from the previous turn and
    a protocol-specific instruction describing how to integrate them
    (e.g. consensus vs. critique). Bias propagation and amplification metrics
    are computed over the trajectory produced by repeated invocations of
    this signature.
    """

    question: str = dspy.InputField()
    context: str = dspy.InputField(desc='optional context')
    options: list[str] = dspy.InputField(desc='ans0, ans1, ans2')
    group: str = dspy.InputField(desc='social group you represent')
    system_prompt: str = dspy.InputField(desc='protocol instructions for this agent')
    peer_answers: str = dspy.InputField(desc='formatted answers from other agents')
    protocol_instruction: str = dspy.InputField(desc='how to respond per protocol')

    answer: str = dspy.OutputField(desc="exactly one of ans0, ans1, ans2 or 'Unknown'")
    reasoning: str = dspy.OutputField(desc='short step-by-step explanation')


class UpdateAnswerWithMemory(dspy.Signature):
    """Interaction-phase signature for memory-augmented agents.

    Identical to ``UpdateAnswer`` plus a ``past_interaction_memory`` slot
    populated by ``Mem0Tools.search``. This is the contract used when the
    memory-mitigation strategy is enabled: it lets the agent revisit
    previously-encountered statements (its own and its peers') to detect
    drift toward a biased consensus. The reasoning field is expected to
    acknowledge the recalled context, which is important for the qualitative
    analysis pass.
    """

    question: str = dspy.InputField()
    context: str = dspy.InputField(desc='optional context')
    options: list[str] = dspy.InputField(desc='ans0, ans1, ans2')
    group: str = dspy.InputField(desc='social group you represent')
    system_prompt: str = dspy.InputField(desc='protocol instructions for this agent')
    peer_answers: str = dspy.InputField(desc='formatted answers from other agents')
    protocol_instruction: str = dspy.InputField(desc='how to respond per protocol')
    past_interaction_memory: str = dspy.InputField(
        desc='relevant recalled past statements from previous rounds'
    )

    answer: str = dspy.OutputField(desc="exactly one of ans0, ans1, ans2 or 'Unknown'")
    reasoning: str = dspy.OutputField(desc='short step-by-step explanation considering past memory')
