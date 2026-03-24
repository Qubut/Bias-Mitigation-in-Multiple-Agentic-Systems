import dspy
from dspy.teleprompt import GEPA

from bias_mitigation.mas.metrics import paper_bias_metrics


def gepa_optimize_mas(program: dspy.Module, trainset: list[dspy.Example]):
    optimizer = GEPA(
        metric=paper_bias_metrics,
        auto='light',
        num_threads=8,
        track_stats=True,
        use_merge=False,
    )
    return optimizer.compile(program, trainset=trainset)
