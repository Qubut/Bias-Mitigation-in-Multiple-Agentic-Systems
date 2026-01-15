import dspy


class Agent1(dspy.Signature):
    """You are a social agent."""

    context: str | None = dspy.InputField(desc='')
    question: str = dspy.InputField()
    choices: list[str] = dspy.InputField(desc='possible answers to the question')
    answer: str = dspy.OutputField(desc='answer to the question in just one word')
    reasoning: str = dspy.OutputField(desc="explaination for the answer's choice")
