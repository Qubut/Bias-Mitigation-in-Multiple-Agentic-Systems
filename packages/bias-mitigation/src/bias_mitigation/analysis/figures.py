"""Paper-grade Matplotlib palette, style, and figure-builder Adapter.

Centralises the publication look-and-feel (cividis sequential / vlag
diverging palette, SIGIR column widths, font sizes) and exposes a small
set of reusable figure builders.  Every builder returns a vanilla
``matplotlib.figure.Figure`` so callers control where and how to save.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from cycler import cycler  # re-exported as ``mpl.cycler`` at runtime but absent from stubs
from scipy import stats as _sstats

from bias_mitigation.analysis.config import ANALYSIS_CONFIG, PaletteConfig

__all__ = [
    'ARM_PALETTE',
    'PAL_METRIC',
    'PAL_SUB',
    'STATE_COLORS',
    'apply_paper_style',
    'arm_bar',
    'asymmetry_bar',
    'forest_plot',
    'hte_heatmap',
    'kaplan_meier_plot',
    'kaplan_meier_with_ci_plot',
    'lifecycle_states_bar',
    'markov_heatmap',
    'mediation_path_diagram',
    'per_category_heatmap',
    'per_model_attribution_bar',
    'posterior_with_rope_plot',
    'power_curve',
    'pvalue_qq_plot',
    'raincloud_plot',
    'recovery_time_histogram',
    'spearman_heatmap',
    'spearman_heatmap_grid',
]


_FIG = ANALYSIS_CONFIG.figures
_DEFAULT_PALETTE: Final[PaletteConfig] = _FIG.palette
_SIGIR_DOUBLE_COLUMN_INCHES: Final[float] = _FIG.sigir_double_column_inches
_SIGIR_SINGLE_COLUMN_INCHES: Final[float] = _FIG.sigir_single_column_inches

# ---------------------------------------------------------------------------
# Register Okabe-Ito derived colormaps so they can be referenced by name in
# analysis.yaml and in imshow calls. Registration happens at import time so
# the names are available before any figure-builder is called.
# oi_sequential : white → Okabe-Ito dark blue   (probability / rate heatmaps)
# oi_diverging  : dark blue ← white → vermillion (effect / correlation heatmaps)
# ---------------------------------------------------------------------------
_OI_SEQ = mpl.colors.LinearSegmentedColormap.from_list(
    'oi_sequential', ['#F7F7F7', '#0072B2'], N=256
)
_OI_DIV = mpl.colors.LinearSegmentedColormap.from_list(
    'oi_diverging', ['#0072B2', '#F7F7F7', '#D55E00'], N=256
)
for _cm in (_OI_SEQ, _OI_DIV):
    if _cm.name not in mpl.colormaps:
        mpl.colormaps.register(_cm)

# ---------------------------------------------------------------------------
# Unifying paper palette (mirrors the legacy ``paper.ipynb`` Nature theme).
# ---------------------------------------------------------------------------
# Single source of truth: ARM_PALETTE derives from the YAML config so that
# both display names (mem0, mem0_opt) and raw pipeline names (mem0g, mem0g_opt)
# resolve to the correct Okabe-Ito colour.  Do NOT hardcode colours here.
ARM_PALETTE: Final[dict[str, str]] = dict(_DEFAULT_PALETTE.arm_colours)
PAL_METRIC: Final[tuple[str, ...]] = ('#4DBBD5', '#00A087', '#E64B35', '#F39B7F')
PAL_SUB: Final[dict[str, str]] = {
    'BBQ': '#3C5488',
    'StereoSet-intra': '#E64B35',
    'StereoSet-inter': '#00A087',
}
# Three lifecycle states — Okabe-Ito palette (colourblind-safe, Wong 2011).
# A: green (never biased = good), B: sky blue (recovered = intermediate),
# C: vermillion (persistent bias = bad).
STATE_COLORS: Final[dict[str, str]] = {
    'A_never_biased': '#009E73',  # Okabe-Ito green
    'B_recovered': '#56B4E9',  # Okabe-Ito sky blue
    'C_persistent': '#D55E00',  # Okabe-Ito vermillion
}


def apply_paper_style(*, palette: PaletteConfig | None = None) -> None:
    """Apply the unifying Nature/SIGIR-style Matplotlib rcParams.

    Mirrors the legacy ``paper.ipynb`` rcParams (publication-grade,
    colourblind-aware) and makes :data:`ARM_PALETTE`,
    :data:`PAL_METRIC`, :data:`STATE_COLORS` available as the single
    chart aesthetic across every notebook.

    Args:
        palette: Optional :class:`PaletteConfig` (kept for backward
            compatibility with the bundled YAML palette).

    Notes:
        Mutates ``matplotlib.rcParams`` globally; intended to be called
        once at the top of a notebook or script.
    """
    p = palette or _DEFAULT_PALETTE
    # Full Okabe-Ito 8-colour cycle for plots that don't use arm-specific colours.
    # Colorblind-safe across deuteranopia, protanopia, tritanopia (Wong 2011,
    # Nature Methods; also natively supported in matplotlib >= 3.11 as 'okabe_ito').
    _okabe_ito = [
        '#0072B2',
        '#E69F00',
        '#009E73',
        '#D55E00',
        '#CC79A7',
        '#56B4E9',
        '#F0E442',
        '#000000',
    ]
    mpl.rcParams.update({
        'figure.figsize': (_SIGIR_SINGLE_COLUMN_INCHES, _SIGIR_SINGLE_COLUMN_INCHES * 0.62),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        # Font stack: Helvetica/Arial preferred; DejaVu Sans guaranteed on Linux.
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        # Math symbols rendered via mathtext; no LaTeX installation required.
        'mathtext.fontset': 'dejavusans',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'axes.titlepad': 7,
        'axes.labelsize': 9,
        'axes.labelpad': 5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.7,
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'axes.prop_cycle': cycler('color', _okabe_ito),
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,
        'grid.color': '#AAAAAA',
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'legend.framealpha': 0.85,
        'legend.edgecolor': '#DDDDDD',
        'legend.borderpad': 0.5,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'patch.linewidth': 0.5,  # subtle bar/patch outline visible in print
        'errorbar.capsize': 3,
        'image.cmap': p.sequential,
        'pdf.fonttype': 42,  # TrueType embedded fonts (ACM/IEEE camera-ready)
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })


def forest_plot(
    estimates: Mapping[str, tuple[float, float, float]],
    *,
    title: str = 'Effect estimates with 95 % CI',
    xlabel: str = 'Effect',
    palette: PaletteConfig | None = None,
) -> mpl.figure.Figure:
    """Build a forest plot of point estimates with 95 % CIs.

    Args:
        estimates: Mapping ``label -> (point, ci_lower, ci_upper)``.
            Display order follows insertion order.
        title: Axis title.
        xlabel: x-axis label.
        palette: Optional :class:`PaletteConfig` providing arm colours.

    Returns:
        The Matplotlib figure (single axis).

    Raises:
        ValueError: If ``estimates`` is empty.
    """
    if not estimates:
        raise ValueError('forest_plot requires at least one estimate')
    p = palette or _DEFAULT_PALETTE
    arm_colour_map = dict(p.arm_colours)
    labels = list(estimates.keys())
    points = np.asarray([estimates[k][0] for k in labels], dtype=np.float64)
    los = np.asarray([estimates[k][1] for k in labels], dtype=np.float64)
    his = np.asarray([estimates[k][2] for k in labels], dtype=np.float64)
    y_positions = np.arange(len(labels))
    fig, ax = plt.subplots(
        figsize=(_SIGIR_SINGLE_COLUMN_INCHES, max(2.0, 0.35 * len(labels) + 0.8)),
    )
    for y, lo, hi, point, label in zip(y_positions, los, his, points, labels, strict=True):
        colour = arm_colour_map.get(label, '#444444')
        ax.plot([lo, hi], [y, y], color=colour, linewidth=2.0)
        ax.plot([point], [y], marker='D', color=colour, markersize=6.0)
    ax.axvline(0.0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def per_category_heatmap(
    matrix: npt.NDArray[np.float64],
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str = 'Per-category effect size',
    cbar_label: str = 'Effect',
    diverging: bool = True,
    palette: PaletteConfig | None = None,
) -> mpl.figure.Figure:
    """Build a per-category heatmap (rows = categories, columns = arms or metrics).

    Args:
        matrix: 2-D array of shape ``(len(row_labels), len(col_labels))``.
        row_labels: Row tick labels.
        col_labels: Column tick labels.
        title: Axis title.
        cbar_label: Colourbar label.
        diverging: ``True`` selects the diverging colormap (centred at 0)
            via ``palette.diverging``; ``False`` uses ``palette.sequential``.
        palette: Optional :class:`PaletteConfig`.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If ``matrix`` shape disagrees with the label lengths.
    """
    if matrix.shape != (len(row_labels), len(col_labels)):
        raise ValueError(
            f'matrix shape {matrix.shape} != (rows={len(row_labels)}, cols={len(col_labels)})',
        )
    p = palette or _DEFAULT_PALETTE
    cmap = p.diverging if diverging else p.sequential
    norm = mpl.colors.TwoSlopeNorm(vcenter=0.0) if diverging else None
    fig, ax = plt.subplots(
        figsize=(_SIGIR_DOUBLE_COLUMN_INCHES, max(2.5, 0.35 * len(row_labels) + 1.5)),
    )
    image = ax.imshow(matrix, aspect='auto', cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return fig


def kaplan_meier_plot(
    curves: Mapping[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    *,
    title: str = 'Survival of "no bias yet" event',
    xlabel: str = 'Turn',
    ylabel: str = 'S(t)',
    palette: PaletteConfig | None = None,
) -> mpl.figure.Figure:
    """Build a step-style Kaplan-Meier plot for one or more arms.

    Args:
        curves: Mapping ``arm_name -> (timeline, survival)`` arrays.
        title: Axis title.
        xlabel: x-axis label.
        ylabel: y-axis label.
        palette: Optional :class:`PaletteConfig` providing arm colours.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If ``curves`` is empty.
    """
    if not curves:
        raise ValueError('kaplan_meier_plot requires at least one curve')
    p = palette or _DEFAULT_PALETTE
    arm_colour_map = dict(p.arm_colours)
    fig, ax = plt.subplots(
        figsize=(_SIGIR_SINGLE_COLUMN_INCHES, _SIGIR_SINGLE_COLUMN_INCHES * 0.62),
    )
    for arm_name, (timeline, survival) in curves.items():
        ax.step(
            timeline,
            survival,
            where='post',
            color=arm_colour_map.get(arm_name, '#444444'),
            label=arm_name,
            linewidth=1.6,
        )
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', frameon=False)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Extended builders (paper iter-7): raincloud, HTE heatmap, Bayesian posterior
# with ROPE, Markov heatmap, per-model attribution, p-value Q-Q, power curve,
# mediation path diagram. Each returns a Matplotlib Figure; PDF/Type-42 is
# enforced by `apply_paper_style()`.
# ----------------------------------------------------------------------------


def raincloud_plot(
    samples: Mapping[str, npt.NDArray[np.float64]],
    *,
    title: str = 'Distribution per arm',
    xlabel: str = 'arm',
    ylabel: str = 'value',
    palette: PaletteConfig | None = None,
) -> mpl.figure.Figure:
    """Raincloud plot (violin + scatter + box) for one or more arms.

    Args:
        samples: Mapping ``arm_name -> 1-D ndarray of values``.
        title: Axis title.
        xlabel: x-axis label.
        ylabel: y-axis label.
        palette: Optional :class:`PaletteConfig`.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If ``samples`` is empty.
    """
    if not samples:
        raise ValueError('raincloud_plot requires at least one sample')
    p = palette or _DEFAULT_PALETTE
    arm_colour_map = dict(p.arm_colours)
    labels = list(samples.keys())
    fig, ax = plt.subplots(
        figsize=(_SIGIR_DOUBLE_COLUMN_INCHES, _SIGIR_SINGLE_COLUMN_INCHES * 0.62),
    )
    positions = np.arange(len(labels))
    rng = np.random.default_rng(0)
    for pos, label in zip(positions, labels, strict=True):
        data = np.asarray(samples[label], dtype=np.float64)
        colour = arm_colour_map.get(label, '#444444')
        v = ax.violinplot([data], positions=[pos], widths=0.7, showextrema=False)
        for body in v['bodies']:  # type: ignore[attr-defined]
            body.set_facecolor(colour)
            body.set_alpha(0.30)
        jitter = rng.normal(loc=pos + 0.30, scale=0.04, size=data.size)
        ax.scatter(jitter, data, s=4.0, alpha=0.35, color=colour, edgecolors='none')
        ax.boxplot(
            [data],
            positions=[pos - 0.30],
            widths=0.18,
            showfliers=False,
            patch_artist=True,
            boxprops={'facecolor': colour, 'alpha': 0.55, 'edgecolor': '#222'},
            medianprops={'color': '#111', 'linewidth': 1.2},
            whiskerprops={'color': '#222'},
            capprops={'color': '#222'},
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def hte_heatmap(
    matrix: npt.NDArray[np.float64],
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    significance_mask: npt.NDArray[np.bool_] | None = None,
    title: str = 'Heterogeneous treatment effects',
    cbar_label: str = 'Mean within-pair difference',
    palette: PaletteConfig | None = None,
) -> mpl.figure.Figure:
    """Heatmap of HTE per (row x column), optionally annotated with significance markers.

    Args:
        matrix: 2-D array of shape ``(len(row_labels), len(col_labels))``.
        row_labels: Row tick labels (e.g. category names).
        col_labels: Column tick labels (e.g. contrasts).
        significance_mask: Optional boolean array of the same shape as
            ``matrix``; cells where ``True`` are annotated with ``*``.
        title: Axis title.
        cbar_label: Colourbar label.
        palette: Optional :class:`PaletteConfig`.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If ``matrix`` shape disagrees with the label lengths or
            ``significance_mask`` shape.
    """
    if matrix.shape != (len(row_labels), len(col_labels)):
        raise ValueError(
            f'matrix shape {matrix.shape} != (rows={len(row_labels)}, cols={len(col_labels)})',
        )
    if significance_mask is not None and significance_mask.shape != matrix.shape:
        raise ValueError('significance_mask shape must match matrix shape')
    p = palette or _DEFAULT_PALETTE
    norm = mpl.colors.TwoSlopeNorm(vcenter=0.0)
    fig, ax = plt.subplots(
        figsize=(_SIGIR_DOUBLE_COLUMN_INCHES, max(2.5, 0.30 * len(row_labels) + 1.5)),
    )
    image = ax.imshow(matrix, aspect='auto', cmap=p.diverging, norm=norm)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    if significance_mask is not None:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if bool(significance_mask[i, j]):
                    ax.text(
                        j,
                        i,
                        '*',
                        ha='center',
                        va='center',
                        color='black',
                        fontsize=10,
                        fontweight='bold',
                    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return fig


def posterior_with_rope_plot(
    *,
    mean: float,
    se: float,
    df: float,
    rope: tuple[float, float] = (-0.02, 0.02),
    title: str = 'Posterior of within-pair difference',
    xlabel: str = 'Δ',
    palette: PaletteConfig | None = None,
) -> mpl.figure.Figure:
    """Approximate Bayesian posterior of a paired difference (Student-t).

    The Region of Practical Equivalence (ROPE) is shaded (Kruschke, 2018).

    Args:
        mean: Posterior mean (the observed within-pair mean difference).
        se: Posterior scale (the within-pair standard error of the mean).
        df: Degrees of freedom (typically ``n - 1``).
        rope: Region of practical equivalence as ``(low, high)``.
        title: Axis title.
        xlabel: x-axis label.
        palette: Optional :class:`PaletteConfig`.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If ``se <= 0``, ``df <= 0`` or ``rope[0] >= rope[1]``.
    """
    if se <= 0.0 or df <= 0.0:
        raise ValueError('se and df must be positive')
    if rope[0] >= rope[1]:
        raise ValueError('rope=(low, high) must satisfy low < high')
    p = palette or _DEFAULT_PALETTE
    width = max(abs(mean) + 6.0 * se, rope[1] * 4.0, 0.05)
    grid = np.linspace(mean - width, mean + width, 800)
    pdf = _sstats.t.pdf(grid, df=df, loc=mean, scale=se)
    fig, ax = plt.subplots(
        figsize=(_SIGIR_SINGLE_COLUMN_INCHES, _SIGIR_SINGLE_COLUMN_INCHES * 0.62),
    )
    ax.plot(grid, pdf, color='#222', linewidth=1.4)
    ax.fill_between(grid, pdf, 0.0, alpha=0.20, color=dict(p.arm_colours).get('mem0', '#1f77b4'))
    ax.axvspan(
        rope[0], rope[1], color='grey', alpha=0.22, label=f'ROPE [{rope[0]:+g}, {rope[1]:+g}]'
    )
    ax.axvline(mean, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('density')
    ax.set_title(title)
    ax.legend(loc='best', frameon=False)
    fig.tight_layout()
    return fig


def markov_heatmap(  # noqa: C901
    matrices: Mapping[str, npt.NDArray[np.float64]],
    *,
    state_labels: Sequence[str],
    title: str = 'Markov transition matrices per arm',
    palette: PaletteConfig | None = None,
) -> mpl.figure.Figure:
    """Grid of row-stochastic Markov transition heatmaps, one panel per arm.

    Four arms are laid out in a 2×2 grid; fewer arms use 1×n.  Each panel
    shows exact cell values as text annotations.  A shared colourbar is placed
    to the right of the grid.

    Args:
        matrices: Mapping ``arm_name -> (k x k) row-stochastic ndarray``.
        state_labels: Labels for the ``k`` states.
        title: Figure suptitle.
        palette: Optional :class:`PaletteConfig`.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If ``matrices`` is empty or any matrix shape disagrees
            with ``state_labels``.
    """
    if not matrices:
        raise ValueError('markov_heatmap requires at least one matrix')
    k = len(state_labels)
    for name, m in matrices.items():
        if m.shape != (k, k):
            raise ValueError(f'matrix {name!r} has shape {m.shape}, expected ({k}, {k})')
    p = palette or _DEFAULT_PALETTE
    n_arms = len(matrices)
    if n_arms <= 2:
        nrows, ncols = 1, n_arms
    elif n_arms == 4:
        nrows, ncols = 2, 2
    else:
        ncols = int(np.ceil(np.sqrt(n_arms)))
        nrows = int(np.ceil(n_arms / ncols))
    _panel = _SIGIR_SINGLE_COLUMN_INCHES * 1.05
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(_SIGIR_DOUBLE_COLUMN_INCHES, nrows * _panel),
        squeeze=False,
    )
    im = None
    for idx, (name, m) in enumerate(matrices.items()):
        ax = axes[idx // ncols, idx % ncols]
        im = ax.imshow(m, aspect='equal', cmap=p.sequential, vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(k))
        ax.set_xticklabels(state_labels, rotation=30, ha='right', fontsize=8)
        ax.set_yticks(np.arange(k))
        ax.set_yticklabels(state_labels, fontsize=8)
        ax.set_xlabel('next state', fontsize=7, labelpad=3)
        ax.set_ylabel('current state', fontsize=7, labelpad=3)
        ax.set_title(name, fontweight='bold', fontsize=9, pad=6)
        for i in range(k):
            for j in range(k):
                ax.text(
                    j,
                    i,
                    f'{m[i, j]:.3f}',
                    ha='center',
                    va='center',
                    color='white' if m[i, j] > 0.6 else '#222222',
                    fontsize=8.5,
                )
    for idx in range(n_arms, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    fig.suptitle(title, fontsize=10, fontweight='bold', y=1.01)
    if im is not None:
        # Reserve space for colourbar on the right without overlapping panels.
        fig.subplots_adjust(right=0.86, hspace=0.50, wspace=0.45)
        cbar_ax = fig.add_axes((0.88, 0.15, 0.025, 0.70))
        fig.colorbar(im, cax=cbar_ax, label='P(next | current)')
    return fig


def per_model_attribution_bar(
    counts: Mapping[str, float],
    *,
    title: str = 'Per-model bias attribution',
    xlabel: str = 'model',
    ylabel: str = 'mean biased-prediction rate',
    palette: PaletteConfig | None = None,
) -> mpl.figure.Figure:
    """Vertical bar chart of per-model bias attribution.

    Args:
        counts: Mapping ``model_name -> rate or count``. Display order
            follows insertion order.
        title: Axis title.
        xlabel: x-axis label.
        ylabel: y-axis label.
        palette: Optional :class:`PaletteConfig`.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If ``counts`` is empty.
    """
    if not counts:
        raise ValueError('per_model_attribution_bar requires at least one model')
    p = palette or _DEFAULT_PALETTE
    labels = list(counts.keys())
    values = np.asarray([counts[k] for k in labels], dtype=np.float64)
    cmap = mpl.colormaps.get_cmap(p.sequential)
    colours = [cmap(0.20 + 0.6 * i / max(1, len(labels) - 1)) for i in range(len(labels))]
    fig, ax = plt.subplots(
        figsize=(_SIGIR_SINGLE_COLUMN_INCHES, _SIGIR_SINGLE_COLUMN_INCHES * 0.62),
    )
    ax.bar(labels, values, color=colours, edgecolor='#222', linewidth=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right')
    fig.tight_layout()
    return fig


def pvalue_qq_plot(
    p_values: npt.NDArray[np.float64],
    *,
    title: str = 'P-value Q-Q (uniform null)',
    palette: PaletteConfig | None = None,
) -> mpl.figure.Figure:
    """Quantile-Quantile plot of p-values against the uniform null.

    Args:
        p_values: 1-D array of p-values in ``[0, 1]``.
        title: Axis title.
        palette: Optional :class:`PaletteConfig`.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If ``p_values`` is empty or contains values outside
            ``[0, 1]``.
    """
    pv = np.asarray(p_values, dtype=np.float64)
    if pv.size == 0:
        raise ValueError('pvalue_qq_plot requires at least one p-value')
    if np.any(pv < 0.0) or np.any(pv > 1.0):
        raise ValueError('p_values must lie in [0, 1]')
    sorted_p = np.sort(pv)
    expected = (np.arange(1, sorted_p.size + 1) - 0.5) / sorted_p.size
    fig, ax = plt.subplots(
        figsize=(_SIGIR_SINGLE_COLUMN_INCHES, _SIGIR_SINGLE_COLUMN_INCHES * 0.62),
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], color='grey', linestyle='--', linewidth=0.8)
    ax.scatter(expected, sorted_p, s=12.0, color='#1f77b4', alpha=0.85, edgecolors='none')
    ax.set_xlabel('expected (uniform)')
    ax.set_ylabel('observed')
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    return fig


def power_curve(
    *,
    n_grid: npt.NDArray[np.int_],
    power: Mapping[str, npt.NDArray[np.float64]],
    title: str = 'Achieved power vs sample size',
    xlabel: str = 'n (paired observations)',
    ylabel: str = '1 - β',
    target_power: float = 0.80,
    palette: PaletteConfig | None = None,
) -> mpl.figure.Figure:
    """Plot one or more power curves vs sample size.

    Args:
        n_grid: 1-D array of sample sizes (the x-axis).
        power: Mapping ``label -> 1-D power array of the same length as
            ``n_grid``.
        title: Axis title.
        xlabel: x-axis label.
        ylabel: y-axis label.
        target_power: Horizontal reference line (default 0.80).
        palette: Optional :class:`PaletteConfig`.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If ``power`` is empty or any series length disagrees
            with ``n_grid``.
    """
    if not power:
        raise ValueError('power_curve requires at least one power series')
    for label, series in power.items():
        if series.shape != n_grid.shape:
            raise ValueError(
                f'power[{label!r}] shape {series.shape} != n_grid shape {n_grid.shape}'
            )
    p = palette or _DEFAULT_PALETTE
    arm_colour_map = dict(p.arm_colours)
    fig, ax = plt.subplots(
        figsize=(_SIGIR_SINGLE_COLUMN_INCHES, _SIGIR_SINGLE_COLUMN_INCHES * 0.62),
    )
    for label, series in power.items():
        ax.plot(
            n_grid, series, label=label, linewidth=1.6, color=arm_colour_map.get(label, '#1f77b4')
        )
    ax.axhline(
        target_power,
        color='grey',
        linestyle='--',
        linewidth=0.8,
        label=f'target = {target_power:.2f}',
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc='best', frameon=False)
    fig.tight_layout()
    return fig


def mediation_path_diagram(
    *,
    a_path: tuple[float, float],
    b_path: tuple[float, float],
    c_prime_path: tuple[float, float],
    indirect: tuple[float, float, float],
    treatment_label: str = 'Treatment',
    mediator_label: str = 'Mediator',
    outcome_label: str = 'Outcome',
    title: str = 'Mediation path diagram',
) -> mpl.figure.Figure:
    """Render a 3-node mediation path diagram with starred significance.

    Each path coefficient is provided as ``(estimate, p_value)``.  Stars
    follow the standard convention: ``***`` for ``p < 0.001``, ``**`` for
    ``p < 0.01``, ``*`` for ``p < 0.05``, blank otherwise.

    Args:
        a_path: ``(a_hat, p)`` for treatment -> mediator.
        b_path: ``(b_hat, p)`` for mediator -> outcome | treatment.
        c_prime_path: ``(c'_hat, p)`` for treatment -> outcome | mediator.
        indirect: ``(estimate, ci_lower, ci_upper)`` of the indirect effect.
        treatment_label: Label for the treatment node.
        mediator_label: Label for the mediator node.
        outcome_label: Label for the outcome node.
        title: Figure title.

    Returns:
        The Matplotlib figure.
    """

    def _stars(p: float) -> str:
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        return ''

    a_hat, a_p = a_path
    b_hat, b_p = b_path
    c_hat, c_p = c_prime_path
    ind_hat, ind_lo, ind_hi = indirect
    fig, ax = plt.subplots(
        figsize=(_SIGIR_DOUBLE_COLUMN_INCHES, _SIGIR_SINGLE_COLUMN_INCHES * 0.62),
    )
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    nodes = {
        treatment_label: (1.5, 1.5),
        mediator_label: (5.0, 5.0),
        outcome_label: (8.5, 1.5),
    }
    for label, (x, y) in nodes.items():
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x - 1.1, y - 0.45),
                2.2,
                0.9,
                boxstyle='round,pad=0.05',
                linewidth=1.0,
                edgecolor='#222',
                facecolor='#eef',
            )
        )
        ax.text(x, y, label, ha='center', va='center', fontsize=9)
    arrows = [
        (treatment_label, mediator_label, f'a = {a_hat:+.3f} {_stars(a_p)}'),
        (mediator_label, outcome_label, f'b = {b_hat:+.3f} {_stars(b_p)}'),
        (treatment_label, outcome_label, f"c' = {c_hat:+.3f} {_stars(c_p)}"),
    ]
    for src, dst, lbl in arrows:
        x0, y0 = nodes[src]
        x1, y1 = nodes[dst]
        ax.annotate(
            '',
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={'arrowstyle': '-|>', 'lw': 1.2, 'color': '#222'},
        )
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.20, lbl, ha='center', va='bottom', fontsize=8)
    ax.text(
        5.0,
        0.4,
        f'indirect a·b = {ind_hat:+.3f}  CI95 [{ind_lo:+.3f}, {ind_hi:+.3f}]',
        ha='center',
        fontsize=8,
        color='#444',
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def kaplan_meier_with_ci_plot(
    curves: Mapping[
        str,
        tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
    ],
    *,
    title: str = 'Kaplan-Meier with 95 % Greenwood CI',
    xlabel: str = 'Turn',
    ylabel: str = 'S(t)',
    palette: PaletteConfig | None = None,
) -> mpl.figure.Figure:
    """KM step plot with a 95 % Greenwood confidence band per arm.

    Args:
        curves: Mapping ``arm_name -> (timeline, survival, ci_lower,
            ci_upper)`` arrays.
        title: Axis title.
        xlabel: x-axis label.
        ylabel: y-axis label.
        palette: Optional :class:`PaletteConfig`.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If ``curves`` is empty.
    """
    if not curves:
        raise ValueError('kaplan_meier_with_ci_plot requires at least one curve')
    p = palette or _DEFAULT_PALETTE
    arm_colour_map = dict(p.arm_colours)
    fig, ax = plt.subplots(
        figsize=(_SIGIR_SINGLE_COLUMN_INCHES, _SIGIR_SINGLE_COLUMN_INCHES * 0.62),
    )
    for arm_name, (timeline, survival, lo, hi) in curves.items():
        colour = arm_colour_map.get(arm_name, '#444444')
        ax.step(timeline, survival, where='post', color=colour, label=arm_name, linewidth=1.6)
        ax.fill_between(timeline, lo, hi, step='post', color=colour, alpha=0.18)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Lifecycle, recovery-time, asymmetry, Spearman builders (paper iter-8).
# ---------------------------------------------------------------------------


def lifecycle_states_bar(
    state_proportions: Mapping[str, Mapping[str, float]],
    *,
    state_order: Sequence[str] = ('A_never_biased', 'B_recovered', 'C_persistent'),
    title: str = 'Bias lifecycle states per arm',
) -> mpl.figure.Figure:
    """Stacked horizontal bar of lifecycle-state proportions per arm.

    Args:
        state_proportions: Mapping ``arm -> {state -> proportion}``.
            Each inner mapping should sum to ~1.0.
        state_order: Stack order left to right (defaults to the
            three-state taxonomy used by :mod:`lifecycle`).
        title: Axis title.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If ``state_proportions`` is empty.
    """
    if not state_proportions:
        raise ValueError('lifecycle_states_bar requires at least one arm')
    arms = list(state_proportions.keys())
    fig, ax = plt.subplots(
        figsize=(_SIGIR_DOUBLE_COLUMN_INCHES, max(2.0, 0.55 * len(arms) + 1.0)),
    )
    y_positions = np.arange(len(arms))
    left = np.zeros(len(arms), dtype=np.float64)
    for state in state_order:
        widths = np.asarray([state_proportions[a].get(state, 0.0) for a in arms], dtype=np.float64)
        ax.barh(
            y_positions,
            widths,
            left=left,
            color=STATE_COLORS.get(state, '#999999'),
            edgecolor='white',
            label=state,
        )
        for y, w, x0 in zip(y_positions, widths, left, strict=True):
            if w > 0.04:
                ax.text(
                    float(x0 + w / 2.0),
                    float(y),
                    f'{w * 100:.1f}%',
                    ha='center',
                    va='center',
                    fontsize=7,
                    color='white' if state != 'B_recovered' else '#222',
                )
        left += widths
    ax.set_yticks(y_positions)
    ax.set_yticklabels(arms)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel('proportion of samples')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.25, linewidth=0.5)
    ax.legend(loc='lower right', ncol=len(state_order), frameon=True)
    fig.tight_layout()
    return fig


def recovery_time_histogram(
    recovery_turn_counts: Mapping[str, Mapping[int, int]],
    *,
    title: str = 'When does bias recovery happen?',
    xlabel: str = 'recovery turn (first turn back to clean)',
    ylabel: str = 'count of recovered samples',
) -> mpl.figure.Figure:
    """Side-by-side bars of recovery-turn counts per arm.

    Args:
        recovery_turn_counts: Mapping ``arm -> {turn_index -> count}``.
            Empty inner maps are skipped (with a printed note).
        title: Axis title.
        xlabel: x-axis label.
        ylabel: y-axis label.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If all arms have empty recovery-turn counts.
    """
    if not recovery_turn_counts or all(not v for v in recovery_turn_counts.values()):
        raise ValueError('recovery_time_histogram requires at least one non-empty arm')
    arms = list(recovery_turn_counts.keys())
    all_turns = sorted({t for c in recovery_turn_counts.values() for t in c})
    fig, ax = plt.subplots(
        figsize=(_SIGIR_DOUBLE_COLUMN_INCHES, _SIGIR_SINGLE_COLUMN_INCHES * 0.65),
    )
    bar_width = 0.8 / max(1, len(arms))
    x_positions = np.arange(len(all_turns), dtype=np.float64)
    for i, arm in enumerate(arms):
        heights = np.asarray(
            [recovery_turn_counts[arm].get(t, 0) for t in all_turns], dtype=np.float64
        )
        offset = (i - (len(arms) - 1) / 2.0) * bar_width
        ax.bar(
            x_positions + offset,
            heights,
            width=bar_width,
            color=ARM_PALETTE.get(arm, '#666666'),
            label=arm,
            edgecolor='white',
            linewidth=0.5,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(t) for t in all_turns])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', frameon=True)
    fig.tight_layout()
    return fig


def arm_bar(
    values: Mapping[str, float],
    *,
    title: str = 'Per-arm metric',
    xlabel: str = 'arm',
    ylabel: str = 'value',
    ylim: tuple[float, float] | None = None,
    fmt: str = '.3f',
    palette: PaletteConfig | None = None,
) -> mpl.figure.Figure:
    """Vertical bar chart with one bar per arm, coloured by the arm palette.

    Designed for scalar summaries such as recovery rate or mean metric value.
    Each bar is annotated with its value above the bar.

    Args:
        values: Mapping ``arm_name -> scalar``.  Display order follows
            insertion order.
        title: Axis title.
        xlabel: x-axis label.
        ylabel: y-axis label.
        ylim: Optional ``(ymin, ymax)`` for the y-axis; default lets
            matplotlib auto-scale with 15 % headroom for the annotation.
        fmt: Format string for the bar-top annotation (e.g. ``'.1%'`` for
            percentages, ``'.3f'`` for raw values).
        palette: Optional :class:`PaletteConfig`.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If ``values`` is empty.
    """
    if not values:
        raise ValueError('arm_bar requires at least one value')
    p = palette or _DEFAULT_PALETTE
    arm_colour_map = dict(p.arm_colours)
    labels = list(values.keys())
    heights = np.asarray([values[k] for k in labels], dtype=np.float64)
    fig, ax = plt.subplots(
        figsize=(_SIGIR_SINGLE_COLUMN_INCHES, _SIGIR_SINGLE_COLUMN_INCHES * 0.65),
    )
    colours = [arm_colour_map.get(lbl, '#666666') for lbl in labels]
    bars = ax.bar(labels, heights, color=colours, edgecolor='white', linewidth=0.5)
    # Resolve effective ymin/ymax; either value in the ylim tuple may be None.
    _ymin_eff = (
        ylim[0] if ylim is not None and ylim[0] is not None else float(np.min(heights)) * 0.95
    )
    _ymax_eff = ylim[1] if ylim is not None and ylim[1] is not None else float(np.max(heights))
    ax.set_ylim(_ymin_eff, _ymax_eff * 1.18)
    annotation_offset = (_ymax_eff - _ymin_eff) * 0.03
    for bar, h in zip(bars, heights):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(h) + annotation_offset,
            format(float(h), fmt),
            ha='center',
            va='bottom',
            fontsize=7,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right')
    fig.tight_layout()
    return fig


def asymmetry_bar(
    counts_by_arm: Mapping[str, Mapping[str, float]],
    *,
    title: str = 'Per-model bias attribution',
    xlabel: str = 'biased-mention count',
) -> mpl.figure.Figure:
    """Grouped horizontal bar of per-model bias counts per arm.

    Annotates the asymmetry index ``(c1-c2)/(c1+c2)`` for the two-model case.

    Args:
        counts_by_arm: Mapping ``arm -> {model_short -> count}``.
        title: Axis title.
        xlabel: x-axis label.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If ``counts_by_arm`` is empty or any inner map empty.
    """
    if not counts_by_arm or any(not v for v in counts_by_arm.values()):
        raise ValueError('asymmetry_bar requires non-empty counts per arm')
    arms = list(counts_by_arm.keys())
    models = sorted({m for v in counts_by_arm.values() for m in v})
    fig, ax = plt.subplots(
        figsize=(_SIGIR_DOUBLE_COLUMN_INCHES, max(2.0, 0.45 * len(models) + 1.0)),
    )
    y_positions = np.arange(len(models), dtype=np.float64)
    bar_width = 0.8 / max(1, len(arms))
    for i, arm in enumerate(arms):
        widths = np.asarray([counts_by_arm[arm].get(m, 0.0) for m in models], dtype=np.float64)
        offset = (i - (len(arms) - 1) / 2.0) * bar_width
        ax.barh(
            y_positions + offset,
            widths,
            height=bar_width,
            color=ARM_PALETTE.get(arm, '#666666'),
            label=arm,
            edgecolor='white',
            linewidth=0.5,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if len(models) == 2:
        notes = []
        for arm in arms:
            c = [counts_by_arm[arm].get(m, 0.0) for m in models]
            tot = c[0] + c[1]
            asym = (c[0] - c[1]) / tot if tot > 0 else float('nan')
            notes.append(f'{arm}: A={asym:+.3f}')
        ax.text(
            0.99,
            0.02,
            '\n'.join(notes),
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=7,
            bbox={'facecolor': 'white', 'edgecolor': '#CCC', 'pad': 4.0},
        )
    ax.legend(loc='best', frameon=True)
    fig.tight_layout()
    return fig


def spearman_heatmap_grid(
    matrices: Mapping[str, npt.NDArray[np.float64]],
    *,
    labels: Sequence[str],
    title: str = 'Spearman correlation per arm',
    palette: PaletteConfig | None = None,
) -> mpl.figure.Figure:
    """Grid of Spearman correlation heatmaps, one panel per arm.

    Four arms are laid out in a 2×2 grid; fewer arms use 1×n.  A single
    shared diverging colourbar runs from −1 to +1.

    Args:
        matrices: Mapping ``arm_name -> square correlation ndarray``.
        labels: Tick labels for both axes (length must equal matrix dimension).
        title: Figure suptitle.
        palette: Optional :class:`PaletteConfig`.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If any matrix is not square or label count mismatches.
    """
    if not matrices:
        raise ValueError('spearman_heatmap_grid requires at least one matrix')
    for arm_name, m in matrices.items():
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError(f'matrix {arm_name!r} is not square: shape={m.shape}')
        if m.shape[0] != len(labels):
            raise ValueError(f'matrix {arm_name!r} dim {m.shape[0]} != len(labels)={len(labels)}')
    p = palette or _DEFAULT_PALETTE
    n_arms = len(matrices)
    if n_arms <= 2:
        nrows, ncols = 1, n_arms
    elif n_arms == 4:
        nrows, ncols = 2, 2
    else:
        ncols = int(np.ceil(np.sqrt(n_arms)))
        nrows = int(np.ceil(n_arms / ncols))
    _panel = _SIGIR_SINGLE_COLUMN_INCHES * 1.05
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(_SIGIR_DOUBLE_COLUMN_INCHES, nrows * _panel),
        squeeze=False,
    )
    im = None
    for idx, (arm_name, matrix) in enumerate(matrices.items()):
        ax = axes[idx // ncols, idx % ncols]
        im = ax.imshow(matrix, cmap=p.diverging, vmin=-1.0, vmax=1.0, aspect='equal')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=7)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(arm_name, fontweight='bold', fontsize=9)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    f'{matrix[i, j]:+.2f}',
                    ha='center',
                    va='center',
                    fontsize=6,
                    color='white' if abs(matrix[i, j]) > 0.5 else 'black',
                )
    for idx in range(n_arms, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    fig.suptitle(title, fontsize=10, fontweight='bold')
    if im is not None:
        fig.subplots_adjust(right=0.86, hspace=0.50, wspace=0.45)
        cbar_ax = fig.add_axes((0.88, 0.15, 0.025, 0.70))
        fig.colorbar(im, cax=cbar_ax, label='Spearman ρ')
    return fig


def spearman_heatmap(
    matrix: npt.NDArray[np.float64],
    *,
    labels: Sequence[str],
    title: str = 'Spearman correlation across primary metrics',
) -> mpl.figure.Figure:
    """Symmetric heatmap of Spearman correlations on a diverging colourmap.

    Args:
        matrix: Square correlation matrix in ``[-1, 1]``.
        labels: Tick labels (length must equal ``matrix.shape[0]``).
        title: Axis title.

    Returns:
        The Matplotlib figure.

    Raises:
        ValueError: If ``matrix`` is not square or label count mismatches.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f'spearman_heatmap expects square matrix; got shape {matrix.shape}')
    if len(labels) != matrix.shape[0]:
        raise ValueError(f'labels length {len(labels)} != matrix dim {matrix.shape[0]}')
    fig, ax = plt.subplots(
        figsize=(_SIGIR_SINGLE_COLUMN_INCHES, _SIGIR_SINGLE_COLUMN_INCHES * 0.95),
    )
    p = _DEFAULT_PALETTE
    im = ax.imshow(matrix, cmap=p.diverging, vmin=-1.0, vmax=1.0, aspect='equal')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f'{matrix[i, j]:+.2f}',
                ha='center',
                va='center',
                fontsize=7,
                color='white' if abs(matrix[i, j]) > 0.5 else 'black',
            )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Spearman rho')
    fig.tight_layout()
    return fig
