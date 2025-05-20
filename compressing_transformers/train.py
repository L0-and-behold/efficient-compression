import os
import time
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from src.CheckpointHandler import CheckpointHandler
from src.Datasets.Wiki40BDataset import ByteWikipediaDataset
from src.Datasets.Wiki40BDataset import WikipediaDatasets
from src.DistributedTransformerTrainer import DistributedTransformerTrainer
from src.Metrics.loss_over_dataset import loss_over_dataset
from src.Metrics.ModelByteSize import ModelByteSize
from src.Metrics.online_code_length import online_code_length_from_dict
from src.Transformer.TransformerConfig import TransformerConfig
from src.DistributedOptimizationProcedures.DrrProcedure import drr_procedure
from src.DistributedOptimizationProcedures.VanillaProcedure import vanilla_procedure
from src.DistributedOptimizationProcedures.Rl1Procedure import rl1_procedure
from src.DistributedOptimizationProcedures.PmmpProcedure import pmmp_procedure

"""Training of a transformer model using distributed data parallelism.

This script handles the complete workflow for training a transformer model on Wikipedia data 
using multiple GPUs with PyTorch's DistributedDataParallel. It includes configuration, 
dataset loading, training execution, and results collection.
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

# Model architecture and pruning parameters
args["alpha"] = 1e-4                          # Regularization strength for ℓ₀-Regularization
args["pmmp"] = True                           # Whether to use  PMMP method
args["initial_p_value"] = 0.7                 # Initial p value for PMMP method
args["beta"] = 10.0                           # Sharpness parameter β for DRR method
args["transformer_config"] = "transformer200k" # Size of transformer model (options: transformer200k, transformer800k, transformer4.1m)
args["training_method"] = rl1_procedure       # Training procedure to use (rl1, vanilla, drr, or pmmp)

# Dataset size and training schedule parameters
args["train_only_on_leading_tokens"] = int(2*4*2048) # Limit training to first N tokens (False or int)
args["epochs_prelude"] = 1                    # Number of epochs for prelude phase (unregularized)
args["epochs"] = 1                            # Number of epochs for main training
args["epochs_fine_tuning"] = 1                # Number of epochs for fine-tuning phase (unregularized with smaller model)
args["stop_epoch_at_batch_prelude"] = False   # Whether to stop prelude epoch early at batch n (False or int)
args["stop_epoch_at_batch"] = False           # Whether to stop main training epoch early at batch n (False or int)
args["stop_epoch_at_batch_fine_tuning"] = False # Whether to stop fine-tuning epoch early at batch n (False or int)
args["batch_size"] = 4                        # Batch size per GPU
args["do_pruning"] = True                     # Whether to perform model pruning
args["first_pruning_after"] = 1               # Epoch after which to start pruning
args["prune_every"] = 1                       # Prune model every n epochs

# Training continuation parameters
args["use_pretrained_model"] = False          # Whether to use a pretrained model
args["use_model_from_experiment"] = None      # Name of experiment to load model from (None or str)
args["use_model_from_run"] = None             # Run ID to load model from (None or str)
args["elapsed_epochs"] = 1                    # Total epochs to be recorded (must match expected total if continuing training)

# Learning and logging parameters
args["learning_rate"] = 1e-5                  # Initial learning rate for optimizer
args["seed"] = 858                            # Random seed for reproducibility
args["tolerated_relative_loss_increase"] = 0.1 # Maximum tolerated loss increase during TAMADE
args["steps_per_chunk"] = 1                   # Number of optimization steps per data chunk
args["log_every"] = 1                         # Log metrics every n optimization steps
args["checkpoint_time"] = 80000               # Save checkpoint every n seconds
args["max_runtime"] = 86000                   # Maximum runtime in seconds before forced termination

# Metrics calculation settings (evaluated after training)
other_settings = {
    "calculate_test_loss": False,             # Whether to calculate mean loss on test set
    "calculate_train_loss": True,             # Whether to calculate mean loss on training set
    "calculate_model_byte_size": True,        # Whether to calculate model size in bytes
    "calculate_non_zero_params": True,        # Whether to count non-zero parameters
    "calculate_on_line_code_length": False,   # Whether to calculate online description length
    "debug": False                            # Whether to print additional debug information
}

################### Functions ###################

def main(args: dict, other_settings: dict, path_to_database: str, experiment_name: str):
    """Main entry point for the training process.
    
    Sets up the distributed environment, initializes the transformer model and trainer,
    and starts the training process.
    
    Args:
        args: Dictionary containing all training and model configuration parameters.
        other_settings: Dictionary with flags for metrics calculation.
        path_to_database: Directory path where results will be stored.
        experiment_name: Name of this experiment for organization and logging.
    """
    # Set the multiprocessing start method to 'spawn' for CUDA compatibility
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    # Create results directory if it doesn't exist
    os.makedirs(path_to_database, exist_ok=True)
    
    # Set random seeds for reproducibility
    set_seed(args["seed"])
    
    # Load transformer configuration based on the selected model size
    transformer_config = TransformerConfig(args["transformer_config"])()
    transformer_config["learning_rate"] = args["learning_rate"]  # Override default learning rate
    args["seq_length"] = transformer_config["seq_length"]  # Store sequence length for dataset preparation
    
    # Initialize the distributed trainer and checkpoint handler
    distributed_trainer = DistributedTransformerTrainer(args, path_to_database, experiment_name, transformer_config, seed=args["seed"])
    checkpointer = CheckpointHandler(experiment_name, args["checkpoint_time"], args["max_runtime"])
    
    # Verify batch size is compatible with distributed training
    assert args["batch_size"] % distributed_trainer.world_size == 0, "Batch size must be divisible by the number of GPUs."
    assert args["batch_size"] >= distributed_trainer.world_size, "Batch size must be larger than the number of GPUs. Otherwise, on-line learning is not realizable with DDP."
    
    # If training on a subset, verify the subset size is compatible with sequence length and batch size
    if args["train_only_on_leading_tokens"]:
        assert args["train_only_on_leading_tokens"] % args["seq_length"] == 0, f"'train_only_on_leading_tokens' must be a multiple of the sequence length {args['seq_length']}"
        assert args["train_only_on_leading_tokens"] % args["batch_size"] == 0, f"'train_only_on_leading_tokens' must be a multiple of the batch size {args['batch_size']}"
    
    print(f"Starting online learning with {distributed_trainer.world_size} GPUs.")
    
    # Start the training process
    train_and_save_results(distributed_trainer, checkpointer, args, other_settings)


def train_and_save_results(distributed_trainer: DistributedTransformerTrainer, checkpointer: CheckpointHandler, args: dict, other_settings: dict, cleanup=True):
    """Execute the complete training workflow including dataset loading, model training, and result saving.
    
    This function handles the entire training pipeline:
    1. Load and prepare datasets
    2. Initialize or restore model and optimizer
    3. Execute the selected training procedure
    4. Calculate metrics and save results
    
    Args:
        distributed_trainer: Manager for distributed training processes.
        checkpointer: Handler for saving and loading model checkpoints.
        args: Dictionary containing training parameters.
        other_settings: Dictionary with flags for which metrics to calculate.
        cleanup: Whether to clean up distributed processes after training. Defaults to True.
    """
    # Get process rank and world size for distributed operations
    rank = distributed_trainer.rank
    world_size = distributed_trainer.world_size
    print(f"Rank {rank}: Using device: {distributed_trainer.device}")
    
    # Load Wikipedia dataset with the configured sequence length
    dataset_local_path = os.path.join(os.getcwd(), 'processed_wiki_dataset.pt')
    datasets = WikipediaDatasets.load_dataset(dataset_local_path, args["seq_length"])
    train_dataset, val_dataset, test_dataset = datasets.train, datasets.validation, datasets.test
    
    # If specified, use only a subset of the training data
    if args["train_only_on_leading_tokens"]:  # overwrite trainset with smaller subset
        train_dataset = WikipediaDatasets.get_subset(0, args["train_only_on_leading_tokens"], train_dataset)
    
    # Create data loaders for each dataset
    dataloaders = [create_dataloader(dataset, world_size, rank, args) for dataset in (train_dataset, val_dataset, test_dataset)]
    dataloader_train, dataloader_val, dataloader_test = dataloaders
    
    # Initialize model and optimizer or load from checkpoint
    model, ddp_model, optimizer = distributed_trainer.model_optimizer()
    model_state_dict, optimizer_state_dict, logs = checkpointer.load_checkpoint()
    
    if model_state_dict:
        # Restore from checkpoint if available
        ddp_model.module.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
    
    # Print model information (only from rank 0 process)
    if rank == 0:
        distributed_trainer.print_model_info(model)
    
    # Begin training
    if rank == 0:
        print("Starting online learning")
    
    t1 = time.time()
    
    # Execute the selected training procedure
    optimization_procedure = args["training_method"]
    new_ddp_model, optimizer, logs, args = optimization_procedure(
        ddp_model, optimizer, logs, distributed_trainer, dataloader_train, val_dataset, checkpointer, args
    )
    
    # Ensure model state is correctly transferred after training
    new_ddp_model.module.load_state_dict(ddp_model.module.state_dict())
    runtime = time.time() - t1
    
    # Calculate metrics and save all results
    save_results(distributed_trainer, logs, ddp_model, optimizer, args, other_settings, dataloaders, runtime)
    
    # Clean up distributed processes if requested
    if cleanup:
        distributed_trainer.cleanup()


def set_seed(seed: int):
    """Set random seeds for reproducibility across all random number generators.
    
    Args:
        seed: Integer seed value for random number generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloader(dataset: ByteWikipediaDataset, world_size: int, rank: int, args: dict) -> DataLoader:
    """Create dataloader with appropriate batch size for distributed training.
    
    Args:
        dataset: Dataset to create loader for.
        world_size: Number of processes in the distributed setup.
        rank: Process rank in the distributed setup.
        args: Dictionary containing configuration parameters including batch size.
    
    Returns:
        DataLoader: Configured dataloader for the dataset.
        
    Raises:
        AssertionError: If batch size is not divisible by world_size or smaller than world_size.
    """
    bs = args["batch_size"]
    assert bs % world_size == 0, "Batch size must be divisible by the number of GPUs."
    assert bs >= world_size, "Batch size must be larger than the number of GPUs. Otherwise, on-line learning is not realizable with DDP."
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)
    return dataloader


def save_results(distributed_trainer: DistributedTransformerTrainer, logs: dict, ddp_model: DDP, optimizer, args: dict, other_settings: dict, dataloaders: list, runtime: float):
    """Save training results, model weights, and metrics to disk.
    
    This function handles the saving of:
    1. Log data as CSV files
    2. Model and optimizer states
    3. Metadata and arguments
    4. Calculated metrics
    
    Only the primary process (rank 0) performs the actual saving.
    
    Args:
        distributed_trainer: The trainer managing distributed training processes.
        logs: Dictionary containing training logs and metrics history.
        ddp_model: The trained DistributedDataParallel model.
        optimizer: The optimizer used for training.
        args: Dictionary of training configuration parameters.
        other_settings: Dictionary with additional settings for metrics and evaluation.
        dataloaders: List of dataloaders for training, validation, and testing.
        runtime: Total training runtime in seconds.
    """
    run, run_df, rank = distributed_trainer.run, distributed_trainer.run_df, distributed_trainer.rank
    model = ddp_model.module
    
    # Calculate various metrics based on settings
    test_loss, train_loss, model_size, non_zero_params, on_line_code_length = calculate_some_metrics(
        distributed_trainer, ddp_model, dataloaders, args, other_settings, logs
    )
    
    # Only rank 0 should save the results to file
    if rank != 0:
        return
    
    # Save training logs as CSV files
    distributed_trainer.save_csv(logs["train_loss_X"], logs["train_loss"], "train_loss")
    distributed_trainer.save_csv(logs["runtime_per_250_batches"], logs["runtime_per_250_batches"], "runtime_per_250_batches")
    distributed_trainer.save_csv(logs["l0_norm_X"], logs["l0_norm"], "l0_norm")
    
    # Save model and optimizer states
    torch.save(model.state_dict(), os.path.join(run.path, "model.pth"))
    torch.save(optimizer.state_dict(), os.path.join(run.path, "optimizer.pth"))
    
    # Log metadata and configuration parameters
    run_df.log_meta()
    run_df.log_args(args)
    
    # Log metrics based on the training logs
    run_df.log_metric("final_train_loss", logs["train_loss"][-1])
    run_df.log_metric("training_runtime", runtime)
    
    # Log optional metrics that were calculated
    if train_loss:
        run_df.log_metric("mean_train_loss", train_loss)
    if test_loss:
        run_df.log_metric("mean_test_loss", test_loss)
    if model_size:
        run_df.log_metric("model_byte_size", model_size)
    if non_zero_params:
        run_df.log_metric("non_zero_params", non_zero_params)
    if on_line_code_length:
        run_df.log_metric("online_description_length_bytes", on_line_code_length)
    
    # Save all metrics to CSV file
    run_df.save_to_CSV()
    return


def calculate_some_metrics(distributed_trainer: DistributedTransformerTrainer, ddp_model: DDP, dataloaders: list, args: dict, other_settings: dict, logs: dict):
    """Calculate various model performance and size metrics based on settings.
    
    Selectively calculates metrics based on the flags in other_settings:
    1. Training and test loss
    2. Model size in bytes
    3. Count of non-zero parameters
    4. Online code length
    
    Args:
        distributed_trainer: The trainer managing distributed processes.
        ddp_model: The trained DistributedDataParallel model.
        dataloaders: List of dataloaders for train, validation, and test datasets.
        args: Dictionary containing training parameters.
        other_settings: Dictionary with flags for which metrics to calculate.
        logs: Dictionary containing training logs.
        
    Returns:
        tuple: Contains (test_loss, train_loss, model_size, non_zero_params, code_length),
               with None for metrics that were not calculated.
    """
    dataloader_train, dataloader_val, dataloader_test = dataloaders
    rank = distributed_trainer.rank
    device = distributed_trainer.device
    model = ddp_model.module
    
    # Calculate training loss if requested
    if other_settings["calculate_train_loss"]:
        print("Calculating train loss")
        train_loss = loss_over_dataset(ddp_model, dataloader_train, args, distributed_trainer, debug=other_settings["debug"])
        print("Train loss: ", train_loss)
    else:
        train_loss = None
    
    # Calculate test loss if requested
    if other_settings["calculate_test_loss"]:
        print("Calculating test loss")
        test_loss = loss_over_dataset(ddp_model, dataloader_test, args, distributed_trainer, debug=other_settings["debug"])
        print("Test loss: ", test_loss)
    else:
        test_loss = None
    
    # Calculate model size in bytes if requested (rank 0 only)
    if other_settings["calculate_model_byte_size"] and rank == 0:
        print("Calculating model size")
        model_size = ModelByteSize.byte_size(model)
        print("Model size: ", model_size)
    else:
        model_size = None
    
    # Count non-zero parameters if requested (rank 0 only)
    if other_settings["calculate_non_zero_params"] and rank == 0:
        print("Calculating non-zero parameters")
        non_zero_params = ModelByteSize.get_model_info(model, device)["non_zero_params"]
        print("Non-zero parameters: ", non_zero_params)
    else:
        non_zero_params = None
    
    # Calculate online code length if requested (rank 0 only)
    if other_settings["calculate_on_line_code_length"] and rank == 0:
        print("Calculating on-line code length")
        code_length = online_code_length_from_dict(logs, args)
        print("On-line code length: ", code_length)
    else:
        code_length = None
    
    return test_loss, train_loss, model_size, non_zero_params, code_length


################### Run ###################

if __name__ == "__main__":
    """Script entry point - executes the main function with the configured parameters.
    
    This conditional ensures the script only runs when executed directly and not when imported.
    """
    main(args, other_settings, path_to_database, experiment_name)