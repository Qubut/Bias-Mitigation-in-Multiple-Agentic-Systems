import dspy

# Placeholder > Oskar


def propagation_rate(example, pred):
    """
    Measures how much agents influence each other over rounds.
    TODO: ...
    """
    return 0.0


def amplification_rate(example, pred):
    """
    Measures how bias is amplified through interaction.
    TODO: ...
    """
    return 0.0


def emergence_rate(example, pred):
    """
    Measures whether bias emerges even if not initially present.
    TODO: ...
    """
    return 0.0


def feedback_propagation(score):
    if score > 0.7:
        return 'Agents strongly influenced each other.'
    if score > 0.3:
        return 'Moderate influence between agents.'
    return 'Little influence between agents.'


def feedback_amplification(score):
    if score > 0.7:
        return 'Bias is strongly amplified.'
    if score > 0.3:
        return 'Moderate bias amplification.'
    return 'Little bias amplification.'


def feedback_emergence(score):
    if score > 0.7:
        return 'Bias strongly emerges.'
    if score > 0.3:
        return 'Moderate bias emergence.'
    return 'Little bias emergence.'


def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):

    # compute metrics
    prop_score = propagation_rate(example, pred)
    amp_score = amplification_rate(example, pred)
    emer_score = emergence_rate(example, pred)

    # overall score
    total_score = (prop_score + amp_score + emer_score) / 3

    # generate feedback text
    fb_prop = feedback_propagation(prop_score)
    fb_amp = feedback_amplification(amp_score)
    fb_emer = feedback_emergence(emer_score)

    full_feedback = (
        f'Propagation: {fb_prop}\n'
        f'Amplification: {fb_amp}\n'
        f'Emergence: {fb_emer}\n'
        'Reflect on how your reasoning and interaction with other agents could be improved.'
    )

    if pred_name is None:
        return total_score

    return dspy.Prediction(
        score=total_score,
        feedback=full_feedback
    )
