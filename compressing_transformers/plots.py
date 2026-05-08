"""
plots.py — Generate loss-vs-size and description-length-vs-alpha plots
from transformer experiment runs.

Usage:
    python plots.py -i input/runs.csv
"""

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt

from src.mdl_analysis.data_loading import load_and_validate, split_data, compute_description_length
from src.mdl_analysis.loss_vs_size_plot import plot_loss_vs_size
from src.mdl_analysis.dl_vs_alpha_plot import plot_dl_vs_alpha
from src.mdl_analysis.dl_vs_size_plot import plot_dl_vs_size
from src.mdl_analysis.constants import label_of_vanilla, label_of_procedure, human_bytes


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Logger:
    """Tee print output to both stdout and a report file."""
    def __init__(self, report_path):
        self.lines = []
        self.report_path = report_path

    def log(self, msg=""):
        print(msg)
        self.lines.append(str(msg))

    def save(self):
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        with open(self.report_path, 'w') as f:
            f.write("\n".join(self.lines) + "\n")
        print(f"\nReport saved to {self.report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate transformer experiment plots')
    parser.add_argument('-i', '--input', required=True, help='Path to runs.csv')
    parser.add_argument('-o', '--output', required=False, help='Path to output plots within output folder', default='')
    parser.add_argument('-y', '--y_max', required=False, help='Maximum test loss-value on the y-axis', default=float('inf'))
    parser.add_argument('-l', '--legend_flag', required=False, help='Legend for plots', default="True")
    parser.add_argument('-t', '--tight_flag', required=False, help='Tight layout plot', default="False")
    parser.add_argument('-nd', '--plots_without_dataset_line', required=False, help='Add dl vs alpha and dl vs size plots without the dataset bar for better visibility if compression is very high', default="False")
    parser.add_argument('--linear-x', action='store_true',
                        help='Use linear x-axis for loss-vs-size plot (default: log)')
    args = parser.parse_args()
    log_x = not args.linear_x
    output_subfolder = args.output
    if output_subfolder != "":
        base_path = 'output/'+output_subfolder
    else:
        base_path = 'output'

    y_max = float(args.y_max)
    if y_max == float('inf'):
        y_suffix = ""
    else:
        y_suffix = "__y_max_" + str(y_max)

    tight_flag = args.tight_flag.lower() == "true"
    if tight_flag:
        tight_suffix = "_tight"
    else:
        tight_suffix = ""

    legend_flag = args.legend_flag.lower() == "true"
    if legend_flag:
        legend_suffix = ""
    else:
        legend_suffix = "_no_legend"

    plots_without_dataset_line = args.plots_without_dataset_line.lower() == "true"


    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    stem = os.path.splitext(os.path.basename(args.input))[0]

    report_path = f"{base_path}/{stem}_{timestamp}{y_suffix}.out"
    logger = Logger(report_path)

    logger.log(f"=== Transformer Plots — {timestamp} ===")
    logger.log(f"Input: {args.input}")

    # Load
    df = load_and_validate(args.input, logger)

    # All runs must share the same dataset size (we plot them on the same axes)
    dataset_sizes = df['train_only_on_leading_tokens'].dropna().unique()
    assert len(dataset_sizes) == 1, f"Expected 1 dataset size, got {len(dataset_sizes)}: {dataset_sizes}"
    dataset_size = float(dataset_sizes[0])
    assert dataset_size > 0, f"Dataset size must be positive, got {dataset_size}"
    logger.log(f"\nDataset size: {dataset_size:.0f} bytes ({dataset_size/1e9:.2f} GB)")

    # Transformer configs
    configs = df['transformer_config'].unique()
    logger.log(f"Transformer configs: {list(configs)}")

    # Split into vanilla baselines (where training_procedure contains 'vanilla_procedure') and regularized procedure runs
    vanilla, procedures = split_data(df, logger)
    assert len(procedures) > 0, "No regularized procedure data found"

    logger.log(f"\n--- Test Loss vs. Model Size ({'linear' if not log_x else 'log'} x) ---")
    fig1 = plot_loss_vs_size(vanilla, procedures, dataset_size, logger, y_max, log_x=log_x, tight_flag=tight_flag, legend_flag=legend_flag)

    logger.log(f"\n--- Description Length vs. α ---")
    fig2 = plot_dl_vs_alpha(vanilla, procedures, dataset_size, logger, tight_flag=tight_flag, legend_flag=legend_flag)

    logger.log(f"\n--- Description Length vs. Model Size ---")
    fig3 = plot_dl_vs_size(vanilla, procedures, dataset_size, logger, tight_flag=tight_flag, legend_flag=legend_flag)

    if plots_without_dataset_line:
        fig22 = plot_dl_vs_alpha(vanilla, procedures, dataset_size, logger, plot_dataset_size=False, tight_flag=tight_flag, legend_flag=legend_flag)
        fig32 = plot_dl_vs_size(vanilla, procedures, dataset_size, logger, plot_dataset_size=False, tight_flag=tight_flag, legend_flag=legend_flag)

    # Save
    os.makedirs('output', exist_ok=True)
    if output_subfolder != "":
        os.makedirs(base_path, exist_ok=True)

    x_suffix = 'linear' if not log_x else 'log'

    fig1.savefig(f'{base_path}/loss_vs_size_{stem}_{x_suffix}{y_suffix}{tight_suffix}{legend_suffix}.pdf', bbox_inches='tight', dpi=300, pad_inches=0)
    fig2.savefig(f'{base_path}/dl_vs_alpha_{stem}{y_suffix}{tight_suffix}{legend_suffix}.pdf', bbox_inches='tight', dpi=300, pad_inches=0)
    fig3.savefig(f'{base_path}/dl_vs_size_{stem}{y_suffix}{tight_suffix}{legend_suffix}.pdf', bbox_inches='tight', dpi=300, pad_inches=0)
    if plots_without_dataset_line:
        fig22.savefig(f'{base_path}/dl_vs_alpha_noDatasetSize_{stem}{y_suffix}{tight_suffix}{legend_suffix}.pdf', bbox_inches='tight', dpi=300, pad_inches=0)
        fig32.savefig(f'{base_path}/dl_vs_size_noDatasetSize_{stem}{y_suffix}{tight_suffix}{legend_suffix}.pdf', bbox_inches='tight', dpi=300, pad_inches=0)

    logger.log(f"\nPlots saved to "+base_path)

    # --- Description Length Summary ---
    logger.log(f"\n--- Description Length Summary (over train set)---")

    logger.log(f"\nVanilla baselines:")
    for _, row in vanilla.iterrows():
        dl = compute_description_length(row['mean_train_loss'], dataset_size, row['model_byte_size'])
        logger.log(f"  {label_of_vanilla(row['non_zero_params'])}: DL = {human_bytes(dl)}  ({dl:.0f} bytes)")

    logger.log(f"\nProcedures (minimum DL per group):")
    for key, sub in procedures.items():
        dls = sub.apply(
            lambda r: compute_description_length(r['mean_train_loss'], dataset_size, r['model_byte_size']),
            axis=1,
        )
        idx_min = dls.idxmin()
        best = sub.loc[idx_min]
        logger.log(f"  {label_of_procedure(key)}: DL = {human_bytes(dls[idx_min])}  ({dls[idx_min]:.0f} bytes)  "
                    f"(α={best['alpha']:.1e}, model={human_bytes(best['model_byte_size'])})")

    logger.save()


if __name__ == '__main__':
    main()
