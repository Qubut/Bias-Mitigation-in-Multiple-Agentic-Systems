"""Render python-statemachine subclasses as MyST mermaid snippets."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import click
from loguru import logger
from statemachine import StateChart

from bias_mitigation.mas.agent_statemachine import AgentStateMachine
from bias_mitigation.mas.mas_statemachine import MASStateMachine
from bias_mitigation.memory.orchestration.statechart import MemoryOrchestrationStateChart
from bias_mitigation.workflows.statechart import WorkflowMachine

if TYPE_CHECKING:
    from statemachine.state import State
    from statemachine.transition import Transition


_MACHINES: tuple[tuple[str, type[StateChart]], ...] = (
    ('mas_statemachine', MASStateMachine),
    ('agent_statemachine', AgentStateMachine),
    ('memory_orchestration_statechart', MemoryOrchestrationStateChart),
    ('workflow_machine', WorkflowMachine),
)

_DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent.parent / 'docs' / '_generated'


def _guard_names(guards: Iterable[object] | None) -> list[str]:
    return [getattr(g, 'name', str(g)) for g in guards or ()]


def _transition_label(t: Transition) -> str:
    cond = _guard_names(getattr(t, 'cond', None))
    unless = _guard_names(getattr(t, 'unless', None))
    return ' · '.join(
        chain(
            [t.event or 'auto'],
            [f'cond {",".join(cond)}'] if cond else [],
            [f'unless {",".join(unless)}'] if unless else [],
        )
    )


def _edges(cls: type[StateChart]) -> Iterable[tuple[State, Transition]]:
    return ((s, t) for s in cls.states for t in s.transitions)


def _initial(cls: type[StateChart]) -> State | None:
    return next((s for s in cls.states if s.initial), None)


def _finals(cls: type[StateChart]) -> Iterable[State]:
    return (s for s in cls.states if s.final)


def to_mermaid(cls: type[StateChart]) -> str:
    initial = _initial(cls)
    body = chain(
        ['stateDiagram-v2'],
        [f'    [*] --> {initial.id}'] if initial else [],
        (
            f'    {s.id} --> {(t.target.id if t.target else s.id)} : {_transition_label(t)}'
            for s, t in _edges(cls)
        ),
        (f'    {s.id} --> [*]' for s in _finals(cls)),
    )
    return '\n'.join(body) + '\n'


def _myst_snippet(diagram: str) -> str:
    return f'```mermaid\n{diagram}```\n'


@click.command()
@click.option(
    '--out-dir',
    type=click.Path(path_type=Path, file_okay=False),
    default=_DEFAULT_OUT_DIR,
    show_default=True,
    help='Directory to write the .md snippets into.',
)
def main(out_dir: Path) -> None:
    """Generate one mermaid `.md` snippet per state machine under OUT_DIR."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, cls in _MACHINES:
        target = out_dir / f'{name}.md'
        target.write_text(_myst_snippet(to_mermaid(cls)), encoding='utf-8')
        logger.info(f'wrote {target}')


if __name__ == '__main__':
    main()
