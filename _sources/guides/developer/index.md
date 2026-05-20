# {octicon}`tools;1em` Developer guide

```{toctree}
:hidden:

maintenance
```

## {octicon}`package;1em` Setup

```bash
uv sync
```

## {octicon}`stack;1em` Packages

| Package | Role |
|---|---|
| `bias_mitigation.data` | Dataset I/O, Pydantic config, splits. |
| `bias_mitigation.mas` | Agent orchestration, protocols, state machines, evaluator, metrics, GEPA. |
| `bias_mitigation.memory` | Async mem0 client, recovery pipeline, orchestrator + statechart. |
| `bias_mitigation.analysis` | polars live-runs pipeline + scipy statistical primitives. |
| `bias_mitigation.workflows` | `WorkflowMachine` for evaluate / train. |
| `containers` + `analysis.containers` | `dependency-injector` wiring. |

## {octicon}`shield-check;1em` Pre-PR checks

::::{tab-set}

:::{tab-item} Lint

```bash
uv run ruff check src/
```
:::

:::{tab-item} Type-check

```bash
uv run mypy src/bias_mitigation
```
:::

:::{tab-item} Docs

```bash
LC_ALL=C.UTF-8 uv run sphinx-build -M html docs docs/_build -W --keep-going
```
:::

:::{tab-item} State diagrams

After editing any `python-statemachine` subclass body:

```bash
uv run generate-statecharts
```
:::

::::

See [Maintenance contract](maintenance.md) for the full set of
code-change → docs-update rules.

## {octicon}`beaker;1em` Testing with DI overrides

Both `Container` (MAS) and `AnalysisContainer` (analysis) expose
`.override()`:

::::{tab-set}

:::{tab-item} MAS container

```python
from dependency_injector import providers
from bias_mitigation.containers import Container

container = Container(mas_config=stub_config)
container.memory_tools.override(providers.Object(stub_mem0_tools))
try:
    program = container.mas_program()
    # ...
finally:
    container.memory_tools.reset_override()
```
:::

:::{tab-item} Analysis container

```python
from dependency_injector import providers
from bias_mitigation.analysis import AnalysisContainer

AnalysisContainer.config.override(providers.Object(stub_analysis_cfg))
try:
    # ...
finally:
    AnalysisContainer.config.reset_override()
```
:::

::::

## {octicon}`law;1em` Standards

- **Docstrings** — imperative, terse. Skip placeholder `Args` /
  `Returns` sections when the signature is obvious.
- **Intervention isolation** — keep intervention-specific logic
  isolated by `InterventionType`; `Container`'s `providers.Selector`
  is the dispatch seam.
- **New state machines** — register in
  `src/scripts/generate_statecharts.py::_MACHINES` so diagrams are
  regenerated on every docs build.
