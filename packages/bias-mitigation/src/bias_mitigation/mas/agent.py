import dspy

from .signatures import InitialAnswer, UpdateAnswer


class Agent:
    def __init__(self, name, group=None):
        self.name = name
        self.group = group

        self.initial = dspy.Predict(InitialAnswer)
        self.update = dspy.Predict(UpdateAnswer)

        self.answer = None
        self.reasoning = None

    def genesis(self, question, choices):

        pred = self.initial(question=question, options=choices)

        self.answer = pred.answer
        self.reasoning = pred.reasoning

        return pred

    def update_answer(self, question, choices, peer_answers):

        pred = self.update(question=question, options=choices, peer_answers=peer_answers)

        self.answer = pred.answer
        self.reasoning = pred.reasoning

        return pred
