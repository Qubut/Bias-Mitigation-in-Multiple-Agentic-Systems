import dspy

from .signatures import InitialAnswer, UpdateAnswer
from ..optimization.metrics import is_biased


class Agent(dspy.Module):
    def __init__(self, name: str, group: str | None = None):
        super().__init__()

        self.name = name
        self.group = group

        self.initial = dspy.ChainOfThought(InitialAnswer)
        self.update = dspy.ChainOfThought(UpdateAnswer)

        self.answer: str | None = None
        self.reasoning: str | None = None
        self.is_biased: bool = False

    def forward(self, question: str, choices: list[str], peer_answers: str | None = None):
        """
        Forward pass for the agent.
        If no peer answers are provided, this is the genesis phase.
        Otherwise, this is an interaction update.
        """

        if peer_answers is None:
            pred = self.initial(
                question=question,
                options=choices,
            )
        else:
            pred = self.update(
                question=question,
                options=choices,
                peer_answers=peer_answers,
            )

        self.answer = pred.answer
        self.reasoning = pred.reasoning

        return pred
