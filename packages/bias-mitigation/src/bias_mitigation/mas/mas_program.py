import dspy

from .agent import Agent
from .system import MultiAgentSystem


class MASProgram(dspy.Module):

    def __init__(self, num_agents=3, rounds=3):
        super().__init__()

        self.num_agents = num_agents
        self.rounds = rounds

    def forward(self, question: str, choices: list[str]):

        agents = [
            Agent(name=f'agent{i}')
            for i in range(self.num_agents)
        ]

        mas = MultiAgentSystem(
            agents=agents,
            rounds=self.rounds
        )

        result = mas.run(question, choices)

        return {
            'answers': result
        }
