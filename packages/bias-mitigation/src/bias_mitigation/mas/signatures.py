import dspy


class InitialAnswer(dspy.Signature):
    """Answer a multiple choice question independently (genesis phase)."""

    question: str = dspy.InputField()
    context: str = dspy.InputField(desc='optional context')
    options: list[str] = dspy.InputField(desc='ans0, ans1, ans2')
    group: str = dspy.InputField(desc='social group you represent')

    answer: str = dspy.OutputField(desc="exactly one of ans0, ans1, ans2 or 'Unknown'")
    reasoning: str = dspy.OutputField(desc='short step-by-step explanation')


class UpdateAnswer(dspy.Signature):
    """Update answer after seeing peer answers (interaction phase)."""

    question: str = dspy.InputField()
    context: str = dspy.InputField(desc='optional context')
    options: list[str] = dspy.InputField(desc='ans0, ans1, ans2')
    group: str = dspy.InputField(desc='social group you represent')
    peer_answers: str = dspy.InputField(desc='formatted answers from other agents')
    protocol_instruction: str = dspy.InputField(desc='how to respond per protocol')

    answer: str = dspy.OutputField(desc="exactly one of ans0, ans1, ans2 or 'Unknown'")
    reasoning: str = dspy.OutputField(desc='short step-by-step explanation')
