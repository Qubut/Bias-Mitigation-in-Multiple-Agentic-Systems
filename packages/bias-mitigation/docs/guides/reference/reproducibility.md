# Reproducibility Checklist

Use this checklist to keep intervention comparisons valid and repeatable.

## Configuration Controls

- [ ] Fix intervention-independent settings (`num_agents`, `rounds`, `protocol`, model configs).
- [ ] Keep dataset sources and transformed schema constant across arms.
- [ ] Keep random seeds fixed for data sampling and split generation.

## Data Pipeline Controls

- [ ] Run scripts in canonical order (`download`, `ingest`, `unify`, `split`).
- [ ] Keep one DB URL consistently across pipeline stages.
- [ ] Version-control config files used for each run.

## Runtime Controls

- [ ] Use matched model endpoints and inference settings across interventions.
- [ ] Reset memory state between test cases in memory-based interventions.
- [ ] Log all runs to MLflow with intervention labels and config parameters.

## Exclusions and Reporting

- [ ] Record technical exclusions with concrete reason codes.
- [ ] Report retained vs excluded counts.
- [ ] Provide sensitivity comparisons when exclusions occur.

## Validation Controls

- [ ] Run lint and tests before evaluation.
- [ ] Build docs with warnings as errors.
- [ ] Ensure diagram and reference pages are up to date for changed behavior.

## Related Pages

- {doc}`/guides/reference/metrics`
- {doc}`/guides/how_to/troubleshooting`
- {doc}`/guides/how_to/scripts`
