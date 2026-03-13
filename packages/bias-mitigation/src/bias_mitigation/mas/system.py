from .protocols import cooperative_protocol


class MultiAgentSystem:
    def __init__(self, agents, rounds=3):
        self.agents = agents
        self.rounds = rounds

    def run(self, question, choices):

        for agent in self.agents:
            agent(question=question, choices=choices)

        for _ in range(self.rounds):
            for agent in self.agents:
                peers = [a for a in self.agents if a != agent]

                context = cooperative_protocol(agent.name, peers)

                agent(
                    question=question,
                    choices=choices,
                    peer_answers=context
                )

        return {a.name: a.answer for a in self.agents}
