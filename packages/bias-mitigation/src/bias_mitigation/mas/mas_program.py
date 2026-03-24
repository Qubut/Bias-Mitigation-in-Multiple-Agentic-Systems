import dspy

from bias_mitigation.data.models.config import MASConfig
from bias_mitigation.data.schemas.datasets import UnifiedBiasEntry
from bias_mitigation.mas.protocols import ProtocolFactory

from .agent import Agent


class MASProgram(dspy.Module):
    """Full declarative DSPy program matching paper appendix."""

    def __init__(self, config: MASConfig):
        super().__init__()
        self.config = config
        self.protocol = ProtocolFactory.get(config.protocol, config.malicious)

    def forward(self, entry: UnifiedBiasEntry) -> dspy.Prediction:
        groups = entry.stereotyped_groups[:self.config.num_agents] or [entry.category] * self.config.num_agents
        agents = [Agent(name=f'agent_{i}', group=groups[i % len(groups)]) for i in range(self.config.num_agents)]

        history: dict[str, list[dspy.Prediction]] = {a.name: [] for a in agents}
        options = [entry.ans0, entry.ans1, entry.ans2]

        # Genesis
        for i, agent in enumerate(agents):
            system_prompt = self.protocol.get_system_prompt(groups[i])
            pred = agent(question=entry.question, context=entry.context, options=options, system_prompt=system_prompt)
            history[agent.name].append(pred)

        for _ in range(self.config.rounds):
            for agent in agents:
                peers = [a for a in agents if a.name != agent.name]
                peer_str = '\n'.join(
                    f'{p.name}: {history[p.name][-1].answer} — {history[p.name][-1].reasoning}'
                    for p in peers
                )
                system_prompt = self.protocol.get_system_prompt(groups[agents.index(agent)])
                pred = agent(
                    question=entry.question,
                    context=entry.context,
                    options=options,
                    system_prompt=system_prompt,
                    peer_answers=peer_str,
                    update_instruction=self.protocol.get_update_instruction(),
                )
                history[agent.name].append(pred)

        return dspy.Prediction(
            history=history,
            final_answers={name: preds[-1].answer for name, preds in history.items()},
            entry_id=entry.id,
        )
