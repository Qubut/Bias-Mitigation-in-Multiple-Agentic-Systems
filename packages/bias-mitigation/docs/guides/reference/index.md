# {octicon}`book;1em` Reference

```{toctree}
:hidden:

architecture
data
metrics
reproducibility
```

The reference section answers **how** and **why** — protocol-level
specifications you can verify against the source.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} {octicon}`workflow;1em` Architecture
:link: architecture
:link-type: doc
:shadow: md

How one sample runs end-to-end. Components, the four
auto-generated state-machine diagrams, DI graph, analysis pipeline.
:::

:::{grid-item-card} {octicon}`database;1em` Data
:link: data
:link-type: doc
:shadow: md

The `download → ingest → unify → split` pipeline.
Submodule responsibilities; reproducibility knobs for splits.
:::

:::{grid-item-card} {octicon}`graph;1em` Metrics
:link: metrics
:link-type: doc
:shadow: md

The four scorers, the GEPA composite, per-metric failure semantics,
and the reporting / stratification layer.
:::

:::{grid-item-card} {octicon}`sync;1em` Reproducibility
:link: reproducibility
:link-type: doc
:shadow: md

Checklist for keeping two runs comparable: config, data pipeline,
runtime, streaming artefacts, exclusions, pre-PR validation.
:::

::::

For class and function signatures, see
[API reference](../../api/index.md).
