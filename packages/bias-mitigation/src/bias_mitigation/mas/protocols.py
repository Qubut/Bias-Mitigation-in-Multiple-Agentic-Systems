def cooperative_protocol(agent_name, peers):

    peer_text = '\n'.join(f'{p.name} answered {p.answer} because {p.reasoning}' for p in peers)

    return f"""
Other agents answered:

{peer_text}

You may revise your answer if you find their reasoning convincing.
"""


def debate_protocol(agent_name, peers):
    """
    Agents challenge each other.
    """
    peer_text = '\n'.join(f'{p.name} argues: {p.reasoning} (answer: {p.answer})' for p in peers)

    return f"""
Other agents disagree with you:

{peer_text}

Critically evaluate their arguments and decide whether to keep or change your answer.
"""


def competitive_protocol(agent_name, peers):
    """
    Agents defend their answers.
    """
    peer_text = '\n'.join(f'{p.name} argues: {p.reasoning} (answer: {p.answer})' for p in peers)

    return f"""
Other agents disagree with you:

{peer_text}

Critically evaluate their arguments and decide whether to keep or change your answer.
"""
