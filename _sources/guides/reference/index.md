# Reference Guide

This guide is the reference hub for experiment design semantics, runtime architecture,
configuration schema, and evaluation variables.

## Sections

```{toctree}
:maxdepth: 1

architecture
data
metrics
reproducibility
```

## Intervention Conditions

- `baseline`: no memory, no prompt optimization.
- `baseline_prompt_opt`: baseline with GEPA-optimized prompts.
- `mem0g`: integrated memory module for retrieval across dialogue turns.
- `mem0g_gepa`: memory module plus GEPA-optimized prompts.

## Core Variables

- **Independent variable**: intervention condition.
- **Primary dependent variable**: propagation rate ($PR_t$).
- **Secondary dependent variables**: emergence rate ($ER_t$), amplification rate ($AR_t$), and system robustness.

## MASConfig Parameters

The `MASConfig` is defined in `config.py` and controls all multi-agent state parameters.

- **`intervention`**: `baseline`, `baseline_prompt_opt`, `mem0g`, or `mem0g_gepa`.
- **`memory_config`**: The nested `Mem0Config` describing vector store params.
- **`agent_models`**: A list of OpenRouter-compatible LLM agent configurations defining each agent's roles and models.
- **`num_agents` / `rounds`**: controls experiment topology (genesis plus interaction rounds).
- **`protocol`**: communication style used for interaction updates.

## Mem0Config Parameters

Found in `memory_config.py`:
- **`llm`**: Setup LLM parameters for Mem0 inference.
- **`embedder`**: Setup Embedding parameters.
- **`vector_store`**: Setup database configurations (e.g. SQLite, Qdrant).

These schema parameters use securely masked Pydantic `SecretStr`s. The MAS Program unwraps them internally before passing to the base Mem0 clients.

## Dataset Scope

- BBQ and StereoSet are used as study benchmarks.
- CrowS-Pairs is excluded from this study design.
- Data processing and sampling are implemented in project data pipelines and should be held fixed across intervention comparisons.

## Analysis Framing

- Compare mean $PR_t$ across intervention conditions.
- Report complementary behavior using $ER_t$, $AR_t$, and robustness metrics.
- Record and disclose technical exclusions to preserve reproducibility.

## Related Pages

- {doc}`/guides/how_to/index`
- {doc}`/api/index`
