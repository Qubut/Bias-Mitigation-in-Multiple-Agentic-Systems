# {octicon}`graph;1em` Analysis notebooks

The three notebooks below reproduce every result reported in
`notebooks/paper/results.tex`. They consume the live-stream rows under
`evaluation/analysis/live/` and the committed dataset splits, and they
render with the outputs already cached in the `.ipynb` (no GPU needed
at docs-build time).

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} {octicon}`number;1em` 01 — Paired main effects
:link: 01_paired_main_effects
:link-type: doc
:shadow: md

Within-pair contrasts for the 2×2 factorial.
**Wilcoxon signed-rank**, Cohen's $d$, Hedges' $g$, **BH-FDR** at
$q = 0.05$ over the 12-cell scoreboard.
:::

:::{grid-item-card} {octicon}`number;1em` 02 — Emergence and survival
:link: 02_emergence_survival
:link-type: doc
:shadow: md

**Kaplan–Meier** survival on $\tau_{\text{first}}$, the 2-state
**Markov chain** on $\{0,1\}$, recovery PMFs, and lifecycle state
mass per arm.
:::

:::{grid-item-card} {octicon}`number;1em` 03 — Sensitivity and confounders
:link: 03_sensitivity_and_confounders
:link-type: doc
:shadow: md

**Manski no-assumption bounds** on the memory effect, per-category
recovery rates, per-model attribution (Llama vs. DeepSeek), and the
full-set vs. paired-set overlay.
:::

::::

```{toctree}
:hidden:

01_paired_main_effects
02_emergence_survival
03_sensitivity_and_confounders
```
