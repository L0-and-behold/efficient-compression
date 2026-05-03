"""Data loading, validation, and splitting for transformer experiment runs.

The CSV contains one row per training run.  Each run's `training_procedure`
column stores the repr() of a Python function, e.g.
  '<function rl1_procedure at 0x14669a1851c0>'
We classify these into short keys (vanilla / rl1 / pmmp / drr) via regex.

Splitting convention:
  - **Vanilla baseline**: vanilla_procedure (plotted as individual crosses)
  - **Procedure runs**: everything else, grouped by procedure key
"""

import os
import re

import numpy as np
import pandas as pd

from src.mdl_analysis.constants import REQUIRED_COLUMNS, PROCEDURE_PATTERNS


def load_and_validate(csv_path: str, logger) -> pd.DataFrame:
    assert os.path.exists(csv_path), f"File not found: {csv_path}"
    df = pd.read_csv(csv_path)
    logger.log(f"Loaded {csv_path}: {len(df)} rows, {len(df.columns)} columns")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"

    return df


def classify_procedure(training_procedure_str: str) -> str:
    """Map the raw training_procedure repr string to a short key.

    Returns 'unknown' if no pattern matches — these rows are excluded
    from all downstream analysis.
    """
    for key, pattern in PROCEDURE_PATTERNS.items():
        if re.search(pattern, str(training_procedure_str)):
            return key
    return 'unknown'


def split_data(df: pd.DataFrame, logger):
    """
    Split the dataframe into:
      - vanilla baselines: vanilla_procedure (individual crosses)
      - procedure runs: everything else, grouped by procedure key
    """
    df = df.copy()
    df['procedure_key'] = df['training_procedure'].apply(classify_procedure)

    # Every row should match a known procedure
    unknown = df[df['procedure_key'] == 'unknown']
    assert len(unknown) == 0, (
        f"{len(unknown)} rows have unrecognised training_procedure: "
        f"{unknown['training_procedure'].unique().tolist()}"
    )

    logger.log(f"\nProcedure breakdown:")
    for key, count in df['procedure_key'].value_counts().items():
        logger.log(f"  {key}: {count} rows")

    # Vanilla baselines: rows classified as 'vanilla'
    vanilla = df[df['procedure_key'] == 'vanilla']
    assert len(vanilla) > 0, "No vanilla baselines found (vanilla_procedure)"
    logger.log(f"\nVanilla baselines: {len(vanilla)} rows")
    for _, row in vanilla.iterrows():
        logger.log(f"  {row['run_id']}  config={row['transformer_config']}  "
                    f"model_bytes={row['model_byte_size']:.0f}  "
                    f"test_loss={row['mean_test_loss']:.6f}")

    # Everything that is not vanilla is a procedure run
    rest = df[~df.index.isin(vanilla.index)]
    procedures = {}
    for key, sub in rest.groupby('procedure_key'):
        # All runs in a procedure group must use the same transformer config
        configs = sub['transformer_config'].unique()
        assert len(configs) == 1, (
            f"Procedure '{key}' has mixed transformer_config values: "
            f"{configs.tolist()}"
        )
        procedures[key] = sub
        alphas = sub['alpha']
        logger.log(f"\n{key}: {len(sub)} runs, α ∈ [{alphas.min():.1e}, {alphas.max():.1e}]")

    return vanilla, procedures


def compute_description_length(mean_loss, dataset_size, model_byte_size):
    """Compute description length in bytes.

    DL = coding_length + model_byte_size
    where coding_length = mean_loss × dataset_size / ln(2) / 8.

    The mean_loss is in nats (natural log base).  We convert to bits (/ln(2))
    then to bytes (/8).  Multiplying by dataset_size gives the total coding
    cost for the dataset.  Adding model_byte_size gives the two-part
    description length: how many bytes to describe the model + the data.
    """
    coding_length = mean_loss * dataset_size / np.log(2) / 8
    return coding_length + model_byte_size

    return df