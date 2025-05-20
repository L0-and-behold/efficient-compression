"""
Field name definitions for experiment logging database.

This module defines the constants that specify field names in the runs.csv output
produced during experiment runs.
"""

META_FIELD_NAMES = [
    "run_id",
    "experiment_name",
    "timestamp",
    "run_time"
]

METRICS_FIELD_NAMES = [
    "final_train_loss",  # Loss evaluated on the last batch
    "final_val_loss",
    "mean_train_loss",  # Mean loss over all batches in the training set
    "mean_test_loss",
    "model_byte_size",
    "non_zero_params",
    "online_description_length_bytes",
    "training_runtime",  # In seconds
]