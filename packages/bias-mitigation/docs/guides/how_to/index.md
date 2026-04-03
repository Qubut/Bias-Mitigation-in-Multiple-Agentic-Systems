# How-To Guide

Step-by-step instructions for executing and extending the preregistered study workflow.

```{toctree}
:maxdepth: 1

scripts
troubleshooting
```

## 1. Run All Intervention Conditions

```bash
uv run train.py --config-path configs/mas_config.yaml --intervention baseline
uv run train.py --config-path configs/mas_config.yaml --intervention baseline_prompt_opt
uv run train.py --config-path configs/mas_config.yaml --intervention mem0g
uv run train.py --config-path configs/mas_config.yaml --intervention mem0g_gepa
```

Use matched configs across runs when comparing outcomes.

## 2. Keep Protocol Settings Comparable

For hypothesis testing, keep these settings stable across intervention runs:

- number of agents,
- number of interaction rounds,
- communication protocol,
- model backends and temperature settings.

## 3. Customize Agent Workflows

To customize an agent debate round:
1. Subclass the base agent defined in `bias_mitigation.mas`.
2. Override the `process_interaction` method.
3. Update `mas_config.yaml` to point to your new module.

## 4. Configure Mem0 Memory Interventions

Enable Mem0 by updating your `memory_config.yaml`:
```yaml
memory_backend: "mem0"
vector_store_path: "./data/vector_store"
```
Re-initialize your agent program. The internal components will automatically read from `bias_mitigation.memory` adapters to embed agent conversations.

## 5. Apply Technical Exclusion Rules

Exclude a test case only for technical failures, such as:

- inference timeout/rate-limit failure,
- malformed output that cannot be mapped to valid options,
- memory backend retrieval failure in memory interventions.

Log exclusion reasons explicitly so summary statistics can be reproduced with and without excluded cases.

## 6. Resolve Mem0 Telemetry Startup Issues

If the program hangs during Mem0 initialization, it's typically related to its telemetry engine ignoring SIGINT. Set the local `MEM0_TELEMETRY` flag:
```bash
MEM0_TELEMETRY=False uv run train.py --intervention mem0g
```
(This is now defaulted to False programmatically within `mem0ai.py`.)

## 7. Prepare Data with Project Scripts

See the dedicated script documentation:

- {doc}`scripts`

## 8. Troubleshooting

For operational issues and common runtime failures, see:

- {doc}`troubleshooting`

## Related Pages

- {doc}`/guides/reference/index`
- {doc}`/api/index`
