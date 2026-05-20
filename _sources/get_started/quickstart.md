# {octicon}`rocket;1em` Quickstart

Runs the four intervention arms used in the study.

:::{dropdown} {octicon}`checklist;1em` Prerequisites
:animate: fade-in-slide-down
:color: primary
:open:

- [uv](https://github.com/astral-sh/uv) — Python package + project
  manager.
- Local model endpoints + API keys per `configs/mas_config.yaml`.
- A writable path for the mem0 vector store (default
  `./.chroma_memories`).
:::

## {octicon}`package;1em` Install

```bash
uv sync
```

## {octicon}`beaker;1em` Train each arm

Run the four arms with the **same** config — only `--intervention`
changes:

::::{tab-set}

:::{tab-item} baseline

```bash
uv run train \
    --config-path configs/mas_config.yaml \
    --intervention baseline
```

Control. No memory, factory prompts.
:::

:::{tab-item} baseline_opt

```bash
uv run train \
    --config-path configs/mas_config.yaml \
    --intervention baseline_opt
```

Isolates the **prompt-optimisation** effect. Writes an optimised
DSPy program to `gepa.save_path`.
:::

:::{tab-item} mem0g

```bash
uv run train \
    --config-path configs/mas_config.yaml \
    --intervention mem0g
```

Isolates the **memory** effect. Mem0-backed recall + store around
each interaction step.
:::

:::{tab-item} mem0g_gepa

```bash
uv run train \
    --config-path configs/mas_config.yaml \
    --intervention mem0g_gepa
```

**Joint treatment.** Tests whether memory + prompt optimisation are
super-additive.
:::

::::

Each run writes:

- {octicon}`file-code;1em` **Optimised DSPy program** at
  `gepa.save_path` — produced by the `*_opt` and `*_gepa` arms only.
- {octicon}`graph;1em` **Live per-sample streams** under
  `evaluation/analysis/live/<run_dir>/` — CSV + JSONL streamed as
  samples complete.
- {octicon}`telescope;1em` **MLflow run** under the configured
  tracking URI, with the intervention tag set.

## {octicon}`graph;1em` Evaluate

```bash
uv run evaluate --config-path configs/mas_config.yaml --subset 1500
```

:::{note}
`--subset` is capped to the dev split size; `min(--subset, len(dev))`
is what's actually evaluated.
:::

## {octicon}`graph;1em` Analyse live runs

```bash
uv run analyze \
    --live-root evaluation/analysis/live \
    --output-root evaluation/analysis/scientific_notebook_outputs \
    --group-by intervention,protocol,dataset_name,model_name \
    --bootstrap-samples 2000
```

| Output | Contents |
|---|---|
| {octicon}`table;1em` `group_estimates.csv` | Tidy long; one row per group × metric with bootstrap CI. |
| {octicon}`code;1em` `group_estimates.json` | Same data, JSON envelope — consumed by the three analysis notebooks. |

CIs are 95 % percentile bootstrap via `scipy.stats.bootstrap`; groups
with fewer than 6 samples degrade to `ci_low == mean == ci_high`.

## {octicon}`workflow;1em` Regenerate state-machine diagrams

```bash
uv run generate-statecharts
```

Reads the four live `python-statemachine` subclasses and writes one
`.md` per machine under `docs/_generated/` (each a MyST snippet with
a fenced `mermaid` block). The architecture page `{include}`-s them;
rerun after any state-machine edit.

## {octicon}`sync;1em` Reproducibility

:::{important}
Keep these knobs fixed across arms in one comparison:
:::

- `num_agents`, `rounds`, `protocol`, `agent_models` — pinned in
  `mas_config.yaml`.
- `gepa.seed` (default `42`) for splits and GEPA.
- For memory arms, memory is reset between test cases at runtime —
  do **not** override `reset_memory_on_genesis: false`.

**Outcomes** — primary is $PR_t$; secondaries are $ER_t$ (emergence,
censored survival), $AR_t$ (amplification), and system robustness.

See [Reproducibility](../guides/reference/reproducibility.md) for the
full checklist.
