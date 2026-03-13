import dspy
import pytest

from bias_mitigation.mas import Agent, MultiAgentSystem


class DummyPredict:
    """
    Mock DSPy predictor so tests don't call a real LLM.
    """

    def __init__(self, answer="A", reasoning="Mock reasoning"):
        self._answer = answer
        self._reasoning = reasoning

    def __call__(self, **kwargs):
        return dspy.Prediction(
            answer=self._answer,
            reasoning=self._reasoning,
        )


@pytest.fixture
def patched_agents(monkeypatch):
    """
    Replace DSPy Predict modules with dummy predictors.
    """

    def fake_predict(signature):
        return DummyPredict()

    monkeypatch.setattr(dspy, "Predict", fake_predict)

    agents = [
        Agent("agent1"),
        Agent("agent2"),
        Agent("agent3"),
    ]

    return agents


def test_mas_runs(patched_agents):
    """
    Ensure the multi-agent system runs and produces answers.
    """

    mas = MultiAgentSystem(agents=patched_agents, rounds=2)

    question = "Who is most likely to enjoy programming?"
    choices = ["A) Engineer", "B) Painter", "C) Unknown"]

    result = mas.run(question=question, choices=choices)

    assert isinstance(result, dict)
    assert len(result) == 3

    for agent_name, answer in result.items():
        assert answer is not None
        assert answer == "A"