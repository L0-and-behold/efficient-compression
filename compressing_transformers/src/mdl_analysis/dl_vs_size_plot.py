"""Plot: Train loss description length vs. model_byte_size.

Shows how total description length (model bytes + coding length) varies
with the model_byte_size.  Vanilla baselines (where training_procedure contains 'vanilla_procedure') and
the raw dataset size are shown as horizontal reference lines.

Only runs with model_byte_size > 0 are plotted (log-x scale requires positive values).
"""

import numpy as np
import matplotlib.pyplot as plt

from src.mdl_analysis.constants import clrs, symbols, marker_size, label_of_procedure, label_of_vanilla, human_bytes, _fmt, PROCEDURE_ORDER
from src.mdl_analysis.data_loading import compute_description_length


def plot_dl_vs_size(vanilla, procedures, dataset_size, logger, plot_dataset_size=True, logscale=True, tight_flag=False, legend_flag=True):
    # assert len(vanilla) > 0, "Need at least one vanilla baseline"
    assert dataset_size > 0, "Dataset size must be positive"

    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    if logscale:
        ax.set_xscale('log')
    ax.set_xlabel('Model Size')
    ax.set_ylabel('Description Length [bytes]')

    size_bounds = [np.inf, -np.inf]  # track range for hline extents
    all_dl = []

    # Procedure order: rl1, drr, pmmp (then any unknown keys alphabetically)
    ordered_keys = [k for k in PROCEDURE_ORDER if k in procedures]
    ordered_keys += sorted(k for k in procedures if k not in PROCEDURE_ORDER)

    for i, key in enumerate(ordered_keys):
        sub = procedures[key]
        color = clrs[i % len(clrs)]
        marker = symbols[i % len(symbols)]

        x = sub['model_byte_size'].values
        y = sub['mean_train_loss'].values

        dl = compute_description_length(y, dataset_size, x)

        mask = x > 0
        if not mask.any():
            continue
        x, dl = x[mask], dl[mask]

        s = np.argsort(x)
        x, dl = x[s], dl[s]

        label = label_of_procedure(key)
        ax.scatter(x, dl, marker=marker, color=color, s=marker_size, label=label)
        ax.plot(x, dl, color=color, linewidth=2, alpha=0.7)

        all_dl.extend(dl)
        size_bounds[0] = min(size_bounds[0], x.min())
        size_bounds[1] = max(size_bounds[1], x.max())

    assert size_bounds[0] < np.inf, "No runs with model_byte_size>0 found — nothing to plot"

    # Vanilla baselines (where training_procedure contains 'vanilla_procedure') as horizontal reference lines,
    # sorted by model size, each with a unique color matching the loss-vs-size plot
    vanilla_sorted = vanilla.sort_values('model_byte_size')
    baseline_color_start = len(ordered_keys)
    for vi, (_, row) in enumerate(vanilla_sorted.iterrows()):
        config_label = label_of_vanilla(row['non_zero_params'])
        baseline_color = clrs[(baseline_color_start + vi) % len(clrs)]
        dl = compute_description_length(row['mean_train_loss'], dataset_size, row['model_byte_size'])
        ax.hlines(y=dl, xmin=size_bounds[0], xmax=size_bounds[1],
                  linestyle='dashed', color=baseline_color, label=config_label)

    # Dataset size = cost of verbatim coding (no model, just store the data)
    if plot_dataset_size:
        ax.hlines(y=dataset_size, xmin=size_bounds[0], xmax=size_bounds[1],
              linestyle='dashed', color='black', label='Dataset Size')

    # correct for possible margin problem on left side
    _, xmax = ax.get_xlim()
    margin = ax.margins()[0]
    if logscale:
        log_padding = 10 ** (margin * (np.log10(xmax) - np.log10(size_bounds[0])))
        ax.set_xlim(left=size_bounds[0] / log_padding)
    else:
        padding = (xmax - size_bounds[0]) * margin / (1 - margin)  # approximate
        ax.set_xlim(left=size_bounds[0] - padding)

    # Human-readable y-axis ticks in MB/GB
    all_dl.append(dataset_size)
    for _, row in vanilla.iterrows():
        all_dl.append(compute_description_length(row['mean_train_loss'], dataset_size, row['model_byte_size']))
    fig.canvas.draw()  # force tick computation
    tick_vals = ax.get_yticks()
    xtick_vals = ax.get_xticks()
    ax.set_yticks(tick_vals, minor=False)
    if tight_flag:
        ax.set_yticklabels([_fmt(t / 1e6, '', False) for t in tick_vals])
        ax.set_xticklabels([_fmt(t / 1e6, '', False) for t in xtick_vals]) # make x axis also have values in MB
    else:
        ax.set_yticklabels([human_bytes(t) for t in tick_vals])
        
    ax.yaxis.set_tick_params(which='minor', size=0)
    ax.yaxis.set_minor_formatter(plt.NullFormatter())

    ylo_lim, yhi_lim = ax.get_ylim()
    ax.set_ylim(max(0.0,ylo_lim), yhi_lim) # Description length is always bigger or equal to 0

    if tight_flag:
        ax.set_xlabel('')
        ax.set_ylabel('')

    if legend_flag:
        ax.legend(loc='best', fontsize=8)

    fig.tight_layout()
    return fig
