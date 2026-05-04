"""
Compute Information Under Curve (IUC) from a single training-loss CSV.

Usage:
    python IUC.py -i input/6160000000.csv

The input CSV must have exactly two columns:
  - Column 1 (no header): iteration number (positive integers, starting at 1)
  - Column 2 (header: any name): training loss in nats (natural-log scale)

The dataset size in bytes is read from the filename stem, e.g.
  6160000000.csv  →  6,160,000,000 bytes

Only the first epoch is used. The epoch boundary is detected where the
iteration counter resets back to 1.

IUC = Σ loss_i / ln(2) / 8   [bytes]   (rectangular integration, nats → bytes)

Output is printed to stdout and saved to output/<stem>_<timestamp>.out
"""

import argparse
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def load_and_validate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=0)

    assert len(df.columns) == 2, (
        f"Expected exactly 2 columns, got {len(df.columns)}: {list(df.columns)}"
    )

    iter_col, loss_col = df.columns[0], df.columns[1]

    assert pd.api.types.is_integer_dtype(df[iter_col]) or df[iter_col].apply(float.is_integer if isinstance(df[iter_col].iloc[0], float) else lambda x: True).all(), \
        f"Column 1 ({iter_col!r}) should contain integer iteration numbers"

    assert pd.api.types.is_float_dtype(df[loss_col]) or pd.api.types.is_numeric_dtype(df[loss_col]), \
        f"Column 2 ({loss_col!r}) should contain numeric loss values"

    assert int(df[iter_col].iloc[0]) == 1, \
        f"First iteration value should be 1, got {df[iter_col].iloc[0]}"

    return df


def dataset_size_from_filename(path: Path) -> int:
    stem = path.stem
    assert stem.isdigit(), (
        f"Filename stem must be a plain integer (dataset bytes), got {stem!r}"
    )
    return int(stem)


def extract_epoch1(df: pd.DataFrame) -> pd.DataFrame:
    iter_vals = df.iloc[:, 0].to_numpy()
    reset_idx = None
    for i in range(1, len(iter_vals)):
        if int(iter_vals[i]) == 1:
            reset_idx = i
            break
    return df.iloc[:reset_idx].copy()


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000:
            return f"{n:,.1f} {unit}"
        n /= 1000
    return f"{n:,.1f} PB"


def build_report(path: Path, df_full: pd.DataFrame, epoch1: pd.DataFrame,
                 dataset_bytes: int, bytes_per_batch: float, iuc: float,
                 timestamp: str) -> str:
    loss_col = df_full.columns[1]
    lines = [
        "=== IUC Report ===",
        f"Timestamp       : {timestamp}",
        "",
        "[File]",
        f"  Input file    : {path}",
        f"  Dataset size  : {dataset_bytes:,} bytes  ({human_bytes(dataset_bytes)})",
        f"  Total rows    : {len(df_full):,}",
        "",
        "[Assertions]",
        f"  ✓ 2 columns: iteration (col 1), {loss_col!r} (col 2)",
        f"  ✓ Filename stem is a valid integer (dataset size in bytes)",
        f"  ✓ Iteration starts at 1",
        "",
        "[Epoch 1]",
        f"  Iterations    : {len(epoch1):,}",
        f"  Epoch 1 ends at row : {len(epoch1):,}" +
        ("  (no reset found — single epoch in file)" if len(epoch1) == len(df_full) else ""),
        "",
        "[Computation]",
        f"  Bytes per batch : {bytes_per_batch:,.2f} bytes / iteration",
        f"  IUC             : {iuc:,.2f} bytes  ({human_bytes(iuc)})",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compute Information Under Curve from a training-loss CSV.")
    parser.add_argument("-i", "--input", required=True, help="Path to input CSV file")
    args = parser.parse_args()

    path = Path(args.input)
    assert path.exists(), f"File not found: {path}"
    assert path.suffix == ".csv", f"Expected a .csv file, got: {path.suffix}"

    # --- Load & validate ---
    df = load_and_validate(path)
    dataset_bytes = dataset_size_from_filename(path)

    # --- Extract first epoch ---
    epoch1 = extract_epoch1(df)
    num_iters = len(epoch1)

    # --- Compute IUC ---
    loss_vals = epoch1.iloc[:, 1].to_numpy(dtype=float)
    bytes_per_batch = dataset_bytes / num_iters
    iuc = float(np.sum(loss_vals) * bytes_per_batch / math.log(2) / 8)

    # --- Build report ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = build_report(path, df, epoch1, dataset_bytes, bytes_per_batch, iuc, timestamp)

    print(report)

    # --- Save to output/ ---
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{path.stem}_{ts_file}.out"
    out_path.write_text(report + "\n")
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
