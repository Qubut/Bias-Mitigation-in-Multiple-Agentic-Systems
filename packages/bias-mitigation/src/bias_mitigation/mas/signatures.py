"""DSPy signatures used by agents for genesis and interaction rounds."""

import dspy


class InitialAnswer(dspy.Signature):
    """Generate an initial independent answer in the genesis phase."""

    question: str = dspy.InputField()
    context: str = dspy.InputField(desc='optional context')
    options: list[str] = dspy.InputField(desc='ans0, ans1, ans2')
    group: str = dspy.InputField(desc='social group you represent')

    answer: str = dspy.OutputField(desc="exactly one of ans0, ans1, ans2 or 'Unknown'")
    reasoning: str = dspy.OutputField(desc='short step-by-step explanation')


class UpdateAnswer(dspy.Signature):
    """Update an answer after observing peer responses."""

    question: str = dspy.InputField()
    context: str = dspy.InputField(desc='optional context')
    options: list[str] = dspy.InputField(desc='ans0, ans1, ans2')
    group: str = dspy.InputField(desc='social group you represent')
    peer_answers: str = dspy.InputField(desc='formatted answers from other agents')
    protocol_instruction: str = dspy.InputField(desc='how to respond per protocol')

    answer: str = dspy.OutputField(desc="exactly one of ans0, ans1, ans2 or 'Unknown'")
    reasoning: str = dspy.OutputField(desc='short step-by-step explanation')


class UpdateAnswerWithMemory(dspy.Signature):
    """Update an answer with peer responses and recalled memory context."""

    question: str = dspy.InputField()
    context: str = dspy.InputField(desc='optional context')
    options: list[str] = dspy.InputField(desc='ans0, ans1, ans2')
    group: str = dspy.InputField(desc='social group you represent')
    peer_answers: str = dspy.InputField(desc='formatted answers from other agents')
    protocol_instruction: str = dspy.InputField(desc='how to respond per protocol')
    past_interaction_memory: str = dspy.InputField(
        desc='relevant recalled past statements from previous rounds'
    )

    answer: str = dspy.OutputField(desc="exactly one of ans0, ans1, ans2 or 'Unknown'")
    reasoning: str = dspy.OutputField(desc='short step-by-step explanation considering past memory')
