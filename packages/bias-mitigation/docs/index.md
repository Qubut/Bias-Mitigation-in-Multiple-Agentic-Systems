# Bias Mitigation Documentation

**Bias Mitigation in MAS** is an experimental framework for measuring and reducing bias propagation in LLM-based multi-agent systems (MAS). The study compares four intervention settings across standardized bias benchmarks and communication protocols.

:::{dropdown} {octicon}`checklist;1em` Key Features
:animate: fade-in-slide-down
:color: primary

- **StateChart Architecture** — Predictable, observable execution replacing ad-hoc agent loops.
- **Modular architecture** — Organized cleanly into three main packages: **bias_mitigation.mas** (Orchestrators, Protocols, Machine), **bias_mitigation.memory** (Mem0-based Context Retrieval), **bias_mitigation.data** (ETL models, BBQ & StereoSet schemas).
- **Extensible Memory** — Easily swap to Mem0, graph memory stores or native LLM-based approaches.
- **Production-ready** — Deeply integrated with enterprise MLflow telemetry, DSPy declarative compilation.
:::

## {octicon}`beaker;1em` Experiment Overview

### Intervention Conditions

- `baseline`: no memory intervention, no prompt optimization.
- `baseline_prompt_opt`: baseline architecture with GEPA-optimized prompts.
- `mem0g`: integrated memory intervention (Mem0-backed retrieval during interaction).
- `mem0g_gepa`: memory intervention combined with GEPA-optimized prompts.

### Hypotheses

- **H0**: no difference in mean propagation rate ($PR_t$) between baseline and memory intervention.
- **H1**: `mem0g` reduces mean $PR_t$ compared with baseline.
- **H2**: `mem0g_gepa` reduces mean $PR_t$ compared with `mem0g`.

### Outcomes

- **Primary**: propagation rate ($PR_t$).
- **Secondary**: emergence rate ($ER_t$), amplification rate ($AR_t$), and system robustness.

### Protocol Defaults

- Two agents per run.
- One genesis phase followed by interaction rounds (default: four rounds).
- Memory is reset between test cases in memory interventions.

### Documentation Architecture

- **Get Started**: practical setup and run commands for experiment execution.
- **Guides**: methodology, architecture, and implementation guidance.
- **API Reference**: module-level technical reference generated from code docstrings.

## {octicon}`package;1em` Installation

To install dependencies and start experimenting:

::::{tab-set}

:::{tab-item} Quick Install

```bash
uv sync
```

:::

:::{tab-item} With Docs

```bash
uv sync --group docs
cd docs
make html
```

:::

::::

## {octicon}`light-bulb;1em` Get Started

::::{grid}

:::{grid-item-card} {octicon}`hourglass;1em` Quickstart
:link: get_started/quickstart
:link-type: doc
:shadow: md

Get started with Bias Mitigation: Configuration, running trials and observing bias interventions.
:::

::::

## {octicon}`book;1em` Guides

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} {octicon}`codescan;1em` Reference Guide
:link: guides/reference/index
:link-type: doc
:shadow: md

Detailed explanations of the internal architecture (MASProgram, Stateful Agents).
:::

:::{grid-item-card} {octicon}`tools;1em` How-To Guide
:link: guides/how_to/index
:link-type: doc
:shadow: md

Step-by-step instructions for integrating Mem0 or customizing execution workflows.
:::

:::{grid-item-card} {octicon}`repo;1em` Developer Guide
:link: guides/developer/index
:link-type: doc
:shadow: md

Contributing guidelines, codebase principles, testing, and debugging setups.
:::

::::

## {octicon}`code-square;1em` API Reference

::::{grid}

:::{grid-item-card} {octicon}`package;1em` API Documentation
:link: api/index
:link-type: doc
:shadow: md

Comprehensive API documentation generated from docstrings.
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

get_started/quickstart
guides/how_to/index
guides/developer/index
guides/reference/index
```

```{toctree}
:caption: API Reference
:hidden:

api/index
```
