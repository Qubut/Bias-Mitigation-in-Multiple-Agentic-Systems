# {octicon}`tools;1em` How-To

```{toctree}
:hidden:

scripts
troubleshooting
```

Recipes for the most common operational tasks. Deeper pages:

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} {octicon}`terminal;1em` Scripts
:link: scripts
:link-type: doc
:shadow: md

The four data-pipeline scripts:
`download → ingest → unify → split`.
:::

:::{grid-item-card} {octicon}`bug;1em` Troubleshooting
:link: troubleshooting
:link-type: doc
:shadow: md

Hangs, pressure events, embedder drift, mem0 telemetry on SIGINT,
Sphinx-build failures.
:::

::::

## {octicon}`beaker;1em` Run the four arms

::::{tab-set}

:::{tab-item} baseline

```bash
uv run train --config-path configs/mas_config.yaml --intervention baseline
```
:::

:::{tab-item} baseline_opt

```bash
uv run train --config-path configs/mas_config.yaml --intervention baseline_opt
```
:::

:::{tab-item} mem0g

```bash
uv run train --config-path configs/mas_config.yaml --intervention mem0g
```
:::

:::{tab-item} mem0g_gepa

```bash
uv run train --config-path configs/mas_config.yaml --intervention mem0g_gepa
```
:::

::::

Use the **same** `mas_config.yaml` across arms; only `--intervention`
changes between runs being compared.

## {octicon}`shield-check;1em` Hold protocol settings comparable

Keep these `mas_config.yaml` fields fixed across arms in a single
comparison:

| Field | Why it must stay fixed |
|---|---|
| `num_agents` | Group size — the per-step communication graph depends on this. |
| `rounds` | Number of interaction rounds after genesis. |
| `protocol` | `cooperative` / `debate` / `competitive` / `malicious`. |
| `agent_models` | Model id, temperature, max_tokens — pin per agent role. |

## {octicon}`plus;1em` Add a new debate protocol

Protocols live in `bias_mitigation.mas.protocols`. Each is a
`ProtocolStrategy` subclass with two methods:
`get_system_prompt(group)` and `get_update_instruction()`.

1. Subclass `ProtocolStrategy`.
2. Register via `ProtocolFactory`.
3. Set `protocol: <new_name>` in the YAML.

## {octicon}`database;1em` Switch the memory backend

`mas_config.yaml::memory_config` is parsed into a `Mem0Config` (see
`bias_mitigation.data.models.memory_config`). The relevant sub-keys
are `llm`, `embedder`, and `vector_store` — each takes a
`provider` + `config` pair compatible with mem0:

```yaml
memory_config:
  llm:
    provider: openai
    config:
      model: meta-llama/Llama-3.1-8B-Instruct
      openai_base_url: http://localhost:30002/v1
      api_key: local
  embedder:
    provider: openai
    config:
      model: mixedbread-ai/mxbai-embed-large-v1
      openai_base_url: http://localhost:30003/v1
      api_key: local
  vector_store:
    provider: chroma
    config:
      path: ./.chroma_memories
      collection_name: bias_mitigation_v2
```

## {octicon}`shield;1em` Disable mem0 telemetry

`MEM0_TELEMETRY=False` is set defensively at import time by
`bias_mitigation.memory.mem0_compat.disable_mem0_telemetry`. The env
var is also respected if set externally:

```bash
MEM0_TELEMETRY=False uv run train --intervention mem0g
```

## {octicon}`x-circle;1em` Exclude a test case

A sample is excluded only on **technical failure**:

- inference timeout / rate-limit exhaustion,
- malformed output that does not map to a valid option,
- memory backend failure under `mem0g` / `mem0g_gepa`.

Failures stream to
`evaluation/analysis/live/<run_dir>/stream_failure_rows.{csv,jsonl}`
with the full traceback in the `error` field
(`mas/evaluator.py::metric`).
