import dspy

from .signatures import InitialAnswer, UpdateAnswer


class Agent(dspy.Module):
    """DSPy Module - per-agent (DeepWiki modular pattern)."""

    def __init__(self, name: str, group: str):
        super().__init__()
        self.name = name
        self.group = group
        self.initial = dspy.ChainOfThought(InitialAnswer)
        self.update = dspy.ChainOfThought(UpdateAnswer)

    def forward(
        self,
        question: str,
        context: str,
        options: list[str],
        system_prompt: str,
        peer_answers: str | None = None,
        update_instruction: str | None = None,
    ) -> dspy.Prediction:
        if peer_answers is None:
            pred = self.initial(
                question=question,
                context=context,
                options=options,
                system_prompt=system_prompt,
                group=self.group,
            )
        else:
            pred = self.update(
                question=question,
                context=context,
                options=options,
                system_prompt=system_prompt,
                peer_answers=peer_answers,
                update_instruction=update_instruction or '',
                group=self.group,
            )

        return dspy.Prediction(
            answer=pred.answer,
            reasoning=pred.reasoning,
            agent_name=self.name,
        )
