"""Plot: Mean test loss vs. model size with description-length iso-lines.

Equipotential curves show lines of constant description length (DL).
From DL = model_bytes + loss × dataset_size / ln(2) / 8, solving for loss:
    loss = 8·ln(2) / dataset_size × (DL − model_bytes)
These are straight lines on the (model_bytes, loss) plane, appearing as
straight lines on a linear-x plot and curves on a log-x plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

from src.mdl_analysis.constants import clrs, symbols, marker_size, label_of_procedure, label_of_vanilla, human_bytes, PROCEDURE_ORDER
from src.mdl_analysis.data_loading import compute_description_length


def equipotential_y(dataset_size, description_length):
    """Return function: model_byte_size → mean_test_loss at fixed DL.

    Derived from:  DL = model_bytes + loss × dataset_size / ln(2) / 8
    Rearranging:   loss = 8·ln(2) / dataset_size × (DL − model_bytes)
    """
    def mean_test_loss(model_byte_size):
        return 8 * np.log(2) / dataset_size * (description_length - model_byte_size)
    return mean_test_loss


def plot_loss_vs_size(vanilla, procedures, dataset_size, logger, log_x=True):
    """Scatter plot of mean test loss vs. model byte size.

    Args:
        vanilla:      DataFrame of vanilla baseline runs (rl1, α=0).
        procedures:   Dict[str, DataFrame] of regularized procedure runs.
        dataset_size: Total dataset size in bytes (for DL computation).
        logger:       Logger instance for reporting.
        log_x:        If True (default), use log scale on x-axis; otherwise linear.
    """
    assert len(vanilla) > 0, "Need at least one vanilla baseline"
    assert len(procedures) > 0, "Need at least one procedure group"
    assert dataset_size > 0, "Dataset size must be positive"

    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    if log_x:
        ax.set_xscale('log')
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: human_bytes(x)))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.set_xlabel('Model Size')
    ax.set_ylabel('Mean Test Loss')

    all_x, all_y, all_dl = [], [], []

    # --- Collect all data first (we need the full DL range to draw iso-lines) ---
    # Procedure order: rl1, drr, pmmp (then any unknown keys alphabetically)
    ordered_keys = [k for k in PROCEDURE_ORDER if k in procedures]
    ordered_keys += sorted(k for k in procedures if k not in PROCEDURE_ORDER)

    proc_plot_data = []
    for i, key in enumerate(ordered_keys):
        sub = procedures[key]
        color = clrs[i % len(clrs)]
        marker = symbols[i % len(symbols)]

        x = sub['model_byte_size'].values
        y = sub['mean_test_loss'].values
        dl = compute_description_length(y, dataset_size, x)

        s = np.argsort(x)
        x, y, dl = x[s], y[s], dl[s]

        label = label_of_procedure(key)
        proc_plot_data.append((key, sub, x, y, dl, s, color, marker, label))

        all_x.extend(x); all_y.extend(y); all_dl.extend(dl)

        for j in range(len(x)):
            logger.log(f"  {key} α={sub['alpha'].iloc[s[j]]:.1e}  "
                        f"model={x[j]/1e6:.1f}MB  loss={y[j]:.6f}  DL={dl[j]/1e6:.1f}MB")

    # Vanilla baselines sorted by model size (smallest first), each with a unique color
    vanilla_sorted = vanilla.sort_values('model_byte_size')
    vanilla_plot_data = []
    # Baselines use colors after the procedure colors
    baseline_color_start = len(ordered_keys)
    for vi, (_, row) in enumerate(vanilla_sorted.iterrows()):
        config_label = label_of_vanilla(row['non_zero_params'])
        x = row['model_byte_size']
        y = row['mean_test_loss']
        dl = compute_description_length(y, dataset_size, x)
        baseline_color = clrs[(baseline_color_start + vi) % len(clrs)]
        vanilla_plot_data.append((x, y, dl, config_label, baseline_color))
        all_x.append(x); all_y.append(y); all_dl.append(dl)

    # --- Equipotential iso-lines (drawn first so data points sit on top) ---
    assert len(all_dl) > 0, "No data points collected"
    all_dl = np.array(all_dl)
    x_lo, x_hi = min(all_x), max(all_x)
    y_lo, y_hi = min(all_y), max(all_y)

    # Iso-lines span slightly beyond the data range (±10%) for visual context
    dl_min, dl_max = all_dl.min(), all_dl.max()
    assert dl_min > 0, f"Description lengths must be positive, got min={dl_min}"
    description_lengths = np.geomspace(dl_min * 0.9, dl_max * 1.1, 8)
    description_lengths = description_lengths[description_lengths > 0]

    norm = LogNorm(vmin=dl_min * 0.9, vmax=dl_max * 1.1)
    cmap = plt.get_cmap('gnuplot')
    if log_x:
        x_vals = np.logspace(np.log10(x_lo) * 0.8, np.log10(x_hi) * 1.2, 200)
    else:
        x_vals = np.linspace(0, x_hi * 1.2, 200)

    for j, dl in enumerate(description_lengths):
        y_vals = equipotential_y(dataset_size, dl)(x_vals)
        color = cmap(norm(dl))
        lbl = '$\\ell$ = const' if j == 3 else None
        ax.plot(x_vals, y_vals, linestyle='dotted', linewidth=1.2, color=color, alpha=0.9, label=lbl)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=[ax], aspect=30, label='Description Length')
    # Human-readable colorbar ticks
    tick_vals = np.geomspace(dl_min * 0.9, dl_max * 1.1, 6)
    nice_ticks = []
    for t in tick_vals:
        if t >= 1e9:
            nice_ticks.append(round(t / 1e8) * 1e8)
        else:
            nice_ticks.append(round(t / 1e7) * 1e7)
    nice_ticks = sorted(set(nice_ticks))
    nice_ticks = [t for t in nice_ticks if dl_min * 0.9 <= t <= dl_max * 1.1]
    cbar.set_ticks(nice_ticks)
    cbar.set_ticklabels([human_bytes(t) for t in nice_ticks])
    cbar.ax.minorticks_off()

    # --- Plot data on top of iso-lines (higher zorder) ---
    for key, sub, x, y, dl, s, color, marker, label in proc_plot_data:
        ax.scatter(x, y, marker=marker, color=color, s=marker_size, label=label, zorder=3)
        ax.plot(x, y, color=color, linewidth=2, alpha=0.7, zorder=3)

    # Vanilla baselines plotted as prominent crosses at highest zorder, each a different color
    for x, y, dl, config_label, bcolor in vanilla_plot_data:
        ax.scatter(x, y, marker='x', s=100, color=bcolor, linewidth=2.5, label=config_label, zorder=5)

    # --- Axis limits ---
    pad = 0.06
    if log_x:
        log_xlo, log_xhi = np.log10(x_lo), np.log10(x_hi)
        ax.set_xlim(10 ** (log_xlo - pad * (log_xhi - log_xlo)),
                    10 ** (log_xhi + pad * (log_xhi - log_xlo)))
    else:
        x_range = x_hi - x_lo
        ax.set_xlim(0, x_hi + pad * x_range)
        # Force a tick at the maximum model size so the full range is visible
        fig.canvas.draw()
        xlo_lim, xhi_lim = ax.get_xlim()
        existing = [t for t in ax.get_xticks() if xlo_lim <= t <= xhi_lim]
        if not any(abs(t - x_hi) < x_range * 0.02 for t in existing):
            ax.set_xticks(sorted(existing + [x_hi]))
    y_pad = 0.05 * (y_hi - y_lo)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    ax.legend(loc='best', fontsize=8)
    return fig
