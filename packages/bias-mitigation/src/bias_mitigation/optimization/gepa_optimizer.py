from dspy import GEPA


def metric_with_feedback(example, prediction, trace=None):
    """
    Generic metric function that uses feedback from the prediction if available.

    If the prediction object has a numeric `feedback` attribute, that value is
    used directly as the metric. Otherwise, a default score of 0.0 is returned.
    """
    feedback = getattr(prediction, "feedback", None)
    if isinstance(feedback, (int, float)):
        return float(feedback)
    return 0.0
def optimize_mas(program, trainset):

    optimizer = GEPA(
        metric=metric_with_feedback,
        auto='light',
        num_threads=8,
        track_stats=True,
        use_merge=False,
    )

    optimized_program = optimizer.compile(
        program,
        trainset=trainset
    )

    return optimized_program
