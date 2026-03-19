import dspy

from .protocols import cooperative_protocol


class MultiAgentSystem:
    def __init__(self, agents, rounds: int = 3):
        self.agents = agents
        self.rounds = rounds

    def run(
        self,
        question: str,
        choices: list[str]
    ) -> dict[str, list[dspy.Prediction]]:
        """
        Returns:
            history: dict[agent_name -> list of dicts with 'answer' and 'reasoning' per turn]
            index = turn number (0 is genesis, 1+ are interaction rounds)
        """

        history: dict[str, list[dspy.Prediction]] = {
            agent.name: [] for agent in self.agents
        }

        #genesis phase
        for agent in self.agents:
            pred = agent(
                question=question,
                choices=choices
            )

            history[agent.name].append(pred)

        #interaction phase
        for round_id in range(self.rounds):
            for agent in self.agents:
                peers = [a for a in self.agents if a != agent]

                context = cooperative_protocol(agent.name, peers)

                pred = agent(
                    question=question,
                    choices=choices,
                    peer_answers=context
                )

                history[agent.name].append(pred)

        return history