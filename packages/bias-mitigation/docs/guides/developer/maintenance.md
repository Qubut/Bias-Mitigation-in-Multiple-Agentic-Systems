# {octicon}`law;1em` Maintenance contract

Edits to the codebase that **require a docs update in the same PR**.

| Code change | Update |
|---|---|
| {octicon}`plus;1em` Add / rename an `InterventionType` member | [`architecture.md`](../reference/architecture.md) component table; rerun `generate-statecharts`. |
| {octicon}`workflow;1em` Edit a `python-statemachine` subclass body | Rerun `generate-statecharts`; commit the diff under `docs/_generated/`. |
| {octicon}`stack;1em` Add / remove a provider on `Container` or `AnalysisContainer` | [`architecture.md`](../reference/architecture.md) DI graph section. |
| {octicon}`graph;1em` Add / rename `_GEPA_METRICS` entries (`mas/metrics.py`) | [`metrics.md`](../reference/metrics.md). |
| {octicon}`code-square;1em` Add / rename a public fn in `analysis/pipeline.py` | [`architecture.md`](../reference/architecture.md) analysis-pipeline section; `analysis/__init__.py` re-exports. |
| {octicon}`gear;1em` Change workflow stages (`WorkflowMachine`) | Rerun the diagram generator; update [`architecture.md`](../reference/architecture.md) workflow section. |

## {octicon}`workflow;1em` State-chart generator

`src/scripts/generate_statecharts.py` walks the four
`python-statemachine` subclasses and writes one `.md` file per
machine into `docs/_generated/`. Each file is a MyST snippet
containing a fenced `mermaid` block, ready for `{include}`.

```bash
uv run generate-statecharts
```

:::{important}
`architecture.md` `{include}`-s these files. **No hand-drawn
mermaid** for state machines — re-run the generator instead.
:::
