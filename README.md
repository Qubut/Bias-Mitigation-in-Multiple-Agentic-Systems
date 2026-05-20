# Bias Mitigation in Multi-Agent Systems

A 2×2 factorial study of bias propagation between cooperating LLM agents
(Llama 3.1-8B Instruct, DeepSeek-R1 distilled into the same 8B backbone)
and of two interventions for reducing it: Mem0-backed vector memory
recalled and stored between turns, and GEPA reflective prompt
optimisation. Items are drawn from BBQ and StereoSet.

The paper, results, and reproducible notebooks live under
`packages/bias-mitigation/notebooks/`. The Sphinx documentation lives under
`packages/bias-mitigation/docs/`.

## The four arms

| Arm | Memory | Prompt |
|---|---|---|
| `baseline` | off | factory |
| `baseline_opt` | off | GEPA |
| `mem0g` | on | factory |
| `mem0g_gepa` | on | GEPA |

## Quick start

```bash
git clone https://github.com/Qubut/Bias-Mitigation-in-Multiple-Agentic-Systems
cd Bias-Mitigation-in-Multiple-Agentic-Systems/packages/bias-mitigation

# Devenv shell (or `direnv allow` if you have direnv).
devenv shell

# Bring up the local SGLang + embedding services.
docker-compose up -d

# Install Python deps.
uv sync

# Train one arm.
uv run train --config-path configs/mas_config.yaml --intervention baseline

# Evaluate on a 1500-item subset.
uv run evaluate --config-path configs/mas_config.yaml --subset 1500

# Analyse the per-sample streaming rows.
uv run analyze \
    --live-root evaluation/analysis/live \
    --output-root evaluation/analysis/scientific_notebook_outputs \
    --group-by intervention,protocol,dataset_name,model_name \
    --bootstrap-samples 2000
```

Swap `--intervention` for `baseline_opt`, `mem0g`, or `mem0g_gepa` to
run the other arms.

## What is measured per sample

| Outcome | Source | Direction |
|---|---|---|
| System robustness | `mas/metrics.py::system_robustness` | ↑ better |
| Emergence rate (first-biased turn) | `mas/metrics.py::emergence_rate` | later better, `-1` = never |
| Propagation rate `PR_t` (primary) | `mas/metrics.py::propagation_rate` | ↓ better |
| Amplification rate | `mas/metrics.py::amplification_rate` | ↓ better |

Per-category breakdowns (gender, race, religion, profession, ...) are
reported alongside the pooled contrasts because both datasets group items
that way.

## Repo layout

```
packages/bias-mitigation/
├── src/
│   ├── bias_mitigation/
│   │   ├── mas/          # MASProgram, agents, state machines, evaluator, metrics, GEPA
│   │   ├── memory/       # mem0.AsyncMemory wrapper, MemoryOrchestrator, contracts
│   │   ├── analysis/     # polars + statsmodels + lifelines analysis pipeline
│   │   ├── data/         # BBQ + StereoSet loaders, splitters, MASConfig
│   │   ├── workflows/    # WorkflowMachine (prepare → build → execute → persist)
│   │   └── containers.py # dependency-injector wiring
│   └── scripts/          # train, evaluate, analyze, dataset CLIs, generate-statecharts
├── configs/              # mas_config.yaml
├── docs/                 # Sphinx + myst-nb site
├── notebooks/            # Quarto config + the three analysis notebooks
├── evaluation/           # checkpoints + per-arm live-stream rows
└── datasets/splits/      # trainset.json, devset.json (seed=42, ratio=0.5)
```

## Stack

`python-statemachine` (lifecycle), `mem0.AsyncMemory` (memory backend),
`tenacity` + `purgatory` (retry + breakers), `dependency-injector`
(DI wiring), DSPy + GEPA (prompt optimisation), `mlflow` (tracking),
`polars` + `scipy.stats.bootstrap` + `statsmodels` + `lifelines` (analysis),
`fairlearn.MetricFrame` (stratified fairness disparities).

## Documentation

Build the Sphinx site locally:

```bash
LC_ALL=C.UTF-8 uv run sphinx-build -M html docs docs/_build
open docs/_build/html/index.html
```

Pages of interest:

- `docs/get_started/quickstart` — full four-arm sequence.
- `docs/guides/reference/architecture` — how one sample runs, with the
  auto-generated state-machine diagrams.
- `docs/guides/reference/metrics` — scorer definitions and the GEPA composite.
- `docs/notebooks/` — the three rendered analysis notebooks.

## License

MIT. See [LICENSE](./LICENSE).
