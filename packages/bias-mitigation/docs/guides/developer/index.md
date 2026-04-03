# Developer Guide

This guide documents implementation constraints for the preregistered MAS bias study.

## Development Priorities

- Preserve intervention comparability across runs.
- Keep randomization and data-processing logic reproducible.
- Avoid introducing condition-specific behavior outside explicitly configured interventions.
- Maintain metric definitions consistent with the preregistration (primary: $PR_t$).

## Setup

```bash
uv sync
```

## Contribution Standards

- Keep public behavior stable unless a change is explicitly scoped.
- Update docs whenever config schema, lifecycle flow, metrics, or CLI changes.
- Add or update tests for changed logic in `data`, `mas`, and `memory` modules.
- Keep intervention-specific logic isolated by `InterventionType`.

## Docstring Standards

- Prefer concise docstrings with explicit `Args`, `Returns`, and `Side Effects` where relevant.
- Avoid placeholder text (`Main module for ...`) in module/class/function docstrings.
- Keep terminology consistent with experiment docs (`baseline`, `baseline_prompt_opt`, `mem0g`, `mem0g_gepa`).

## Local Quality Checks

Run these checks before opening a PR:

```bash
uv run ruff check .
uv run ruff format --check .
uv run sphinx-build -M html docs docs/_build -W --keep-going
```

## Pull Request Checklist

- [ ] Updated related docs pages and API references.
- [ ] Updated prompt tracker todos where requested.
- [ ] Ran lint/tests/docs checks locally.
- [ ] Added migration notes for config or schema changes.

## System Structure

The project relies on:
- **DSPy** for language model programming.
- **MLflow** for tracing and experiment tracking.
- **Mem0** for long-term memory retrieval within memory interventions.

Core package boundaries:

- `bias_mitigation.data`: dataset transformation, split tracking, reproducibility artifacts.
- `bias_mitigation.mas`: agent orchestration, protocols, state machine lifecycle.
- `bias_mitigation.memory`: memory retrieval implementation and memory backend integration.

## Intervention Modes

Supported intervention labels:

- `baseline`
- `baseline_prompt_opt`
- `mem0g`
- `mem0g_gepa`

When adding new functionality, ensure behavior changes are isolated by intervention type and do not leak into baseline execution.
