import dspy


class InitialAnswer(dspy.Signature):
    """Answer a multiple choice question independently."""

    question: str = dspy.InputField()
    options: list[str] = dspy.InputField(desc='possible answers')

    answer: str = dspy.OutputField(desc='choose one option')
    reasoning: str = dspy.OutputField(desc='short explanation')


class UpdateAnswer(dspy.Signature):
    """Update answer after seeing other agents."""

    question: str = dspy.InputField()
    options: list[str] = dspy.InputField(desc='possible answers')
    peer_answers: str = dspy.InputField(desc='answers from other agents')

    answer: str = dspy.OutputField(desc='choose one option')
    reasoning: str = dspy.OutputField(desc='short explanation')
