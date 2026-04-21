"""Plot: Description length vs. regularization parameter α.

Shows how total description length (model bytes + coding length) varies
with the regularization strength α.  Vanilla baselines (rl1, α=0) and
the raw dataset size are shown as horizontal reference lines.

Only runs with α > 0 are plotted (log-x scale requires positive values).
"""

import numpy as np
import matplotlib.pyplot as plt

from src.mdl_analysis.constants import clrs, symbols, marker_size, label_of_procedure, label_of_config, human_bytes, PROCEDURE_ORDER
from src.mdl_analysis.data_loading import compute_description_length


def plot_dl_vs_alpha(vanilla, procedures, dataset_size, logger):
    assert len(vanilla) > 0, "Need at least one vanilla baseline"
    assert dataset_size > 0, "Dataset size must be positive"

    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    ax.set_xscale('log')  # α is always plotted on log scale
    ax.set_xlabel('α')
    ax.set_ylabel('Description Length [bytes]')

    alpha_bounds = [np.inf, -np.inf]  # track range for hline extents
    all_dl = []

    # Procedure order: rl1, drr, pmmp (then any unknown keys alphabetically)
    ordered_keys = [k for k in PROCEDURE_ORDER if k in procedures]
    ordered_keys += sorted(k for k in procedures if k not in PROCEDURE_ORDER)

    for i, key in enumerate(ordered_keys):
        sub = procedures[key]
        color = clrs[i % len(clrs)]
        marker = symbols[i % len(symbols)]

        alpha = sub['alpha'].values
        x = sub['model_byte_size'].values
        y = sub['mean_test_loss'].values
        dl = compute_description_length(y, dataset_size, x)

        # Only runs with α > 0 can appear on a log-x scale.
        # α=0 runs are either vanilla baselines or PMMP references.
        mask = alpha > 0
        if not mask.any():
            continue
        alpha, dl = alpha[mask], dl[mask]

        s = np.argsort(alpha)
        alpha, dl = alpha[s], dl[s]

        label = label_of_procedure(key)
        ax.scatter(alpha, dl, marker=marker, color=color, s=marker_size, label=label)
        ax.plot(alpha, dl, color=color, linewidth=2, alpha=0.7)

        all_dl.extend(dl)
        alpha_bounds[0] = min(alpha_bounds[0], alpha.min())
        alpha_bounds[1] = max(alpha_bounds[1], alpha.max())

    assert alpha_bounds[0] < np.inf, "No runs with α>0 found — nothing to plot"

    # Vanilla baselines (rl1 with α=0) as horizontal reference lines,
    # sorted by model size, each with a unique color matching the loss-vs-size plot
    vanilla_sorted = vanilla.sort_values('model_byte_size')
    baseline_color_start = len(ordered_keys)
    for vi, (_, row) in enumerate(vanilla_sorted.iterrows()):
        config_label = label_of_config(row['transformer_config'])
        baseline_color = clrs[(baseline_color_start + vi) % len(clrs)]
        dl = compute_description_length(row['mean_test_loss'], dataset_size, row['model_byte_size'])
        ax.hlines(y=dl, xmin=alpha_bounds[0], xmax=alpha_bounds[1],
                  linestyle='dashed', color=baseline_color, label=config_label)

    # Dataset size = cost of verbatim coding (no model, just store the data)
    ax.hlines(y=dataset_size, xmin=alpha_bounds[0], xmax=alpha_bounds[1],
              linestyle='dashed', color='black', label='Dataset Size')

    # Human-readable y-axis ticks in MB/GB
    all_dl.append(dataset_size)
    for _, row in vanilla.iterrows():
        all_dl.append(compute_description_length(row['mean_test_loss'], dataset_size, row['model_byte_size']))
    fig.canvas.draw()  # force tick computation
    tick_vals = ax.get_yticks()
    ax.set_yticks(tick_vals, minor=False)
    ax.set_yticklabels([human_bytes(t) for t in tick_vals])
    ax.yaxis.set_tick_params(which='minor', size=0)
    ax.yaxis.set_minor_formatter(plt.NullFormatter())

    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    return fig
