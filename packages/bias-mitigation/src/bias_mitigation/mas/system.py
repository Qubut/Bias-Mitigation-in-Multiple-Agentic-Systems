import dspy

from .agent import Agent
from .protocols import cooperative_protocol


class MultiAgentSystem:
    def __init__(
        self,
        agents: list[Agent],
        rounds: int = 3,
        protocol=cooperative_protocol
    ):
        self.agents = agents
        self.rounds = rounds
        self.protocol = protocol

    def run(
        self,
        question: str,
        choices: list[str]
    ) -> dict[str, list[dspy.Prediction]]:
        """
        Returns:
            history: dict[agent_name -> list of Predictions per turn]
        """
        history: dict[str, list[dspy.Prediction]] = {
            agent.name: [] for agent in self.agents
        }

        # genesis phase
        for agent in self.agents:
            pred = agent(
                question=question,
                choices=choices
            )
            history[agent.name].append(pred)

        # interaction phase
        for _ in range(self.rounds):
            for agent in self.agents:
                peers = [a for a in self.agents if a != agent]

                context = self.protocol(agent.name, peers)

                pred = agent(
                    question=question,
                    choices=choices,
                    peer_answers=context
                )

                history[agent.name].append(pred)

        return history
