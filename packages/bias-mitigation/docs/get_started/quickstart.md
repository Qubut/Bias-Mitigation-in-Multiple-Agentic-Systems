# Quickstart

This quickstart runs the preregistered intervention conditions for the MAS bias propagation study.

## Prerequisites

- [uv](https://github.com/astral-sh/uv)
- Configured local model endpoints and API keys expected by `configs/mas_config.yaml`

## Install

```bash
uv sync
```

## Run Preregistered Intervention Arms

```bash
# 1) Baseline
uv run train.py --config-path configs/mas_config.yaml --intervention baseline

# 2) Baseline + prompt optimization
uv run train.py --config-path configs/mas_config.yaml --intervention baseline_prompt_opt

# 3) Memory intervention
uv run train.py --config-path configs/mas_config.yaml --intervention mem0g

# 4) Memory + prompt optimization
uv run train.py --config-path configs/mas_config.yaml --intervention mem0g_gepa
```

## Notes for Preregistration Fidelity

- Keep the same dataset splits and randomization seed across intervention runs.
- Use the same protocol and model settings when comparing intervention effects.
- For memory interventions, ensure memory is reset between test cases (handled in runtime flow).
- Track all runs in MLflow and compare primary outcome ($PR_t$) plus secondary outcomes ($ER_t$, $AR_t$, robustness).

## Related Pages

- {doc}`/guides/how_to/index`
- {doc}`/guides/reference/index`
- {doc}`/api/index`
