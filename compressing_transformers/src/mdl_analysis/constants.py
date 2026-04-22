"""Shared constants and formatting helpers for transformer experiment plots."""

import re

# Plot styling
symbols = ['o', 's', '^', 'h', '<', 'v', 'p', '>', '*', 'd', '8', 'D', 'P', 'X']

clrs = [
    '#0b846c',  # Light Green
    '#fec47c',  # Light Orange
    '#fe6989',  # Light Red
    '#125b9a',  # Light Blue
    "#9467bd",  # Medium Purple
    "#7f7f7f",  # Gray
    "#e377c2",  # Pink  
    "#8c564b",  # Chestnut Brown
    "#d62728",  # Brick Red
    "#2ca02c",  # Forest Green
    "#bcbd22",  # Olive
    "#17becf"   # Light Blue
]

figsize = (8, 3.5)
marker_size = 13

# Desired legend order: rl1 first, then drr, then pmmp
PROCEDURE_ORDER = ['rl1', 'drr', 'pmmp']

# Known procedure keys — must match PROCEDURE_PATTERNS
KNOWN_PROCEDURES = set(PROCEDURE_ORDER) | {'vanilla'}

# Map procedure function repr strings to short keys via regex.
# The CSV stores Python repr() strings like '<function rl1_procedure at 0x...>'.
PROCEDURE_PATTERNS = {
    'vanilla': r'vanilla_procedure',
    'rl1':  r'rl1_procedure',
    'pmmp': r'pmmp_procedure',
    'drr':  r'drr_procedure',
}

# Columns required in the input CSV
REQUIRED_COLUMNS = [
    'training_procedure', 'model_byte_size', 'mean_test_loss', 'mean_train_loss',
    'alpha', 'train_only_on_leading_tokens', 'transformer_config',
    'non_zero_params',
]


def label_of_procedure(procedure):
    """Map short procedure key to display label.

    Procedure keys come from classify_procedure() in data_loading.py:
      'rl1'  → R-ℓ₁
      'drr'  → DRR
      'pmmp' → PMMP
    Unknown keys are uppercased as a fallback.
    """
    if procedure == 'rl1':
        return "R-ℓ₁"
    elif procedure == 'drr':
        return "DRR"
    elif procedure == 'pmmp':
        return "PMMP"
    else:
        return procedure.upper()


def label_of_vanilla(non_zero_params) -> str:
    """Format vanilla baseline label from actual parameter count, rounded to millions."""
    millions = round(float(non_zero_params) / 1e6)
    return f"UT {millions}M"


# Kept for backwards compatibility but no longer used for vanilla labels.
def label_of_config(config: str) -> str:
    """Parse transformer config name to 'UT <size>M' label."""
    m = re.match(r't(\d+)_(\d+)p', config)
    if m:
        return f"UT {m.group(1)}M"
    m = re.match(r't(\d+)p(\d+)_(\d+)p', config)
    if m:
        return f"UT {m.group(1)}.{m.group(2)}M"
    return config


def human_bytes(value):
    """Format a byte count as human-readable string with 3 significant digits."""
    def _fmt(v, unit):
        if v >= 100:
            return f"{v:.0f}{unit}"
        elif v >= 10:
            s = f"{v:.1f}"
        else:
            s = f"{v:.2f}"
        # Strip trailing zero after decimal point (10.0 -> 10, 1.50 -> 1.5)
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return f"{s}{unit}"

    if abs(value) >= 1e9:
        return _fmt(value / 1e9, 'GB')
    elif abs(value) >= 1e6:
        return _fmt(value / 1e6, 'MB')
    elif abs(value) >= 1e3:
        return _fmt(value / 1e3, 'kB')
    return f"{value:.0f}B"