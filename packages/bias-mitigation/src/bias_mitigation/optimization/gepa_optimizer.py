from dspy import GEPA

from .metrics import metric_with_feedback


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
