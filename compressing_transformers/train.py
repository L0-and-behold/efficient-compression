import os, sys

_file_location = os.path.dirname(__file__)
sys.path.append(_file_location)
sys.path.append(os.path.abspath(os.path.join(_file_location, '..')))

from src.TrainFunctions import TrainFunctions
from src.DistributedOptimizationProcedures.DrrProcedure import drr_procedure
from src.DistributedOptimizationProcedures.VanillaProcedure import vanilla_procedure
from src.DistributedOptimizationProcedures.Rl1Procedure import rl1_procedure
from src.DistributedOptimizationProcedures.PmmpProcedure import pmmp_procedure
from src.Datasets.Wiki40BDataset import ByteWikipediaDataset
from src.Datasets.Wiki40BDataset import WikipediaDatasets # has to be imported here despite not being used for torch.load to unpickle the loaded dataset correctly within the function called below.

"""Training of a transformer model using distributed data parallelism.

This script handles the workflow for training a transformer model on Wikipedia data 
using multiple GPUs with PyTorch's DistributedDataParallel. It provides an interface for configuration of all necessary parameters and uses the TrainFunctions class to execute dataset loading, training, and results collection.
"""

###################
# Training Settings
###################

"""
Configuration section - defines all parameters needed for the training run.

See README.md for more information.

The configuration is divided into several groups:
1. Basic paths and experiment identification
2. Model architecture and learning procedure parameters
3. Dataset and training logistics parameters
4. Checkpointing and runtime parameters
5. Metrics calculation settings
"""

# Base paths and experiment identification
path_to_database = os.path.join(os.getcwd(), "experiment-results")
experiment_name = "example-experiment"
args = {}

### Regularization Parameters - BEGIN
# Model architecture and pruning parameters
args["alpha"] = 1e-4                          # Regularization strength for ℓ₀-Regularization
args["initial_p_value"] = 0.7                 # Initial p value for PMMP method
args["initial_u_value"] = 3.0                 # Initial u value for PMMP method
args["beta"] = 10.0                           # Sharpness parameter β for DRR method
### Regularization Parameters - END

### Model Configuration - BEGIN
args["training_method"] = pmmp_procedure       # Training procedure to use (rl1, vanilla, drr, or pmmp)
args["transformer_config"] = "transformer200k" # Transformer model
### Model Configuration - END

### Optimizer Parameters - BEGIN
args["learning_rate"] = 3e-4                  # Initial learning rate for optimizer
args["AdamW_betas"] = (0.9, 0.95)             # The beta parameters for the AdamW optimizer (typical choices include (0.9, 0.95) or (0.9, 0.999))
args["warmup_steps"] = 2000                   # Warmup increases the learning rate linearly to `learning_rate` in `warmup_steps` steps
args["weight_decay"] = 0.01                   # Weight decay applies L2 regularization to all parameters except biases, LayerNorm-weights and `u` and `p` parameters of PMMP
### Optimizer Parameters - END

### Training Process Parameters - BEGIN
args["iterations_per_epoch"] = 2              # the number of iterations or batches which are processed per epoch. Each iteration, `batch_size` many `seq_length` long batches are processed. Therefore, make sure the dataset contains at least `args["iterations_per_epoch"]*batch_size*seq_length` many tokens (where `seq_length` is the context window of the model).
args["tokens_per_epoch"] = False             # (False or int). If not False, then ["iterations_per_epoch"] is overwritten and set equal to int(args["tokens_per_epoch"] / batch_size / seq_length), where seq_length is the context window of the model. Make sure args["tokens_per_epoch"] is not bigger than the number of tokens in your dataset. This parameter can also be used to make the training token number per epoch equal to N times the number of parameters of the model.
args["epochs_prelude"] = 0                    # Number of epochs for prelude phase (unregularized)
args["epochs"] = 1                            # Number of epochs for main training
args["epochs_fine_tuning"] = 1                # Number of epochs for fine-tuning phase (unregularized with smaller model)
args["stop_epoch_at_batch_prelude"] = False   # Whether to stop prelude epoch early at batch n (False or int)
args["stop_epoch_at_batch"] = False           # Whether to stop main training epoch early at batch n (False or int)
args["stop_epoch_at_batch_fine_tuning"] = int(0.15*args["iterations_per_epoch"]) # Whether to stop fine-tuning epoch early at batch n (False or int)
### Training Process Parameters - END


### Pruning Parameters - BEGIN
args["do_pruning"] = True                     # Whether to perform model pruning
args["first_pruning_after"] = 1               # Epoch after which to start pruning
args["prune_every"] = 1                       # Prune model every n epochs
### Pruning Parameters - END


### Model Loading Parameters - BEGIN
args["use_pretrained_model"] = False          # Whether to use a pretrained model
args["use_model_from_experiment"] = None      # Name of experiment to load model from (None or str)
args["use_model_from_run"] = None             # Run ID to load model from (None or str)
args["elapsed_epochs"] = 1                    # Total epochs to be recorded (must match expected total if continuing training)
### Model Loading Parameters - END

### General Training Parameters - BEGIN
args["seed"] = 858                              # Random seed for reproducibility
args["tolerated_relative_loss_increase"] = 0.01 # Maximum tolerated loss increase during TAMADE
args["steps_per_chunk"] = 1                     # Number of optimization steps per data chunk
args["log_every"] = 1                           # Log metrics every n optimization steps
args["checkpoint_time"] = 80000                 # Save checkpoint every n seconds
args["max_runtime"] = 86000                     # Maximum runtime in seconds before forced termination
### General Training Parameters - END

### Metrics Calculation Parameters - BEGIN
# (evaluated after training)
other_settings = {
    "calculate_test_loss": False,             # Whether to calculate mean loss on test set
    "calculate_train_loss": True,             # Whether to calculate mean loss on training set
    "calculate_model_byte_size": True,        # Whether to calculate model size in bytes
    "calculate_non_zero_params": True,        # Whether to count non-zero parameters
    "calculate_on_line_code_length": False,   # Whether to calculate online description length
    "debug": False                            # Whether to print additional debug information
}
args["only_process_every_nth_batch_when_calculating_train_loss"] = 4      # Must be a natural number. Only every args["only_process_every_nth_batch_when_calculating_train_loss"] of the train set is evaluated for computing the train loss. Numbers bigger 1 increase speed but make the estimate less accurate.
args["only_process_every_nth_batch_when_calculating_test_loss"] = 4      # Must be a natural number. Only every args["only_process_every_nth_batch_when_calculating_test_loss"] of the test set is evaluated for computing the test loss. Numbers bigger 1 increase speed but make the estimate less accurate.
### Metrics Calculation Parameters - END


################### Run ###################

if __name__ == "__main__":
    """Script entry point - executes the main function with the configured parameters.
    
    This conditional ensures the script only runs when executed directly and not when imported.
    """
    trainfs = TrainFunctions(args, other_settings, path_to_database, experiment_name)
    trainfs.train_and_save_results()