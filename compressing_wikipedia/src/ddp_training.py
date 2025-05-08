import os
import time
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import numpy as np
import random
from torch.nn.parallel import DistributedDataParallel as DDP

from src.Transformer.TransformerConfig import TransformerConfig

from src.Datasets.Wiki40BDataset import ByteWikipediaDataset
from src.Datasets.Wiki40BDataset import WikipediaDatasets
from src.DistributedTransformerTrainer import DistributedTransformerTrainer
from src.CheckpointHandler import CheckpointHandler

from src.Metrics.loss_over_dataset import loss_over_dataset
from src.Metrics.ModelByteSize import ModelByteSize
from src.Metrics.online_code_length import online_code_length_from_dict

from src.DistributedOptimizationProcedures.VanillaProcedure import vanilla_procedure
from src.DistributedOptimizationProcedures.DrrProcedure import drr_procedure

"""
Functions and specification for training a model using distributed data parallelism. 
See `train.py` for an example of how to use these functions.

Assumes that `other_settings` and `args` are dictionaries with keys as in `train.py`. 
Assumes that started in ddp mode with torch.multiprocessing.spawn
    >>> if mp.get_start_method(allow_none=True) != 'spawn':
    >>>     mp.set_start_method('spawn', force=True)
"""

def train_and_save_results(distributed_trainer: DistributedTransformerTrainer, checkpointer: CheckpointHandler, args: dict, other_settings: dict, cleanup=True):
    """
    Execute complete training workflow with checkpointing and result saving.
    
    Handles dataset loading, model training, evaluation and metric calculation.
    """
    rank = distributed_trainer.rank
    world_size = distributed_trainer.world_size
    print(f"Rank {rank}: Using device: {distributed_trainer.device}")

    # Load Wikipedia dataset
    dataset_local_path = os.path.join(os.getcwd(), 'processed_wiki_dataset.pt')
    datasets = WikipediaDatasets.load_dataset(dataset_local_path, args["seq_length"])
    train_dataset, val_dataset, test_dataset = datasets.train, datasets.validation, datasets.test
    if args["train_only_on_leading_tokens"]: # overwrite trainset with smaller subset
        train_dataset = WikipediaDatasets.get_subset(0, args["train_only_on_leading_tokens"], train_dataset)
    dataloaders = [create_dataloader(dataset, world_size, rank, args) for dataset in (train_dataset, val_dataset, test_dataset)]
    dataloader_train, dataloader_val, dataloader_test = dataloaders

    # Load model and optimizer
    model, ddp_model, optimizer = distributed_trainer.model_optimizer()
    model_state_dict, optimizer_state_dict, logs = checkpointer.load_checkpoint()
    if model_state_dict:
        ddp_model.module.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
    if rank == 0:
        distributed_trainer.print_model_info(model)

    # Train model
    if rank == 0:
        print("Starting online learning")

    t1 = time.time()

    optimization_procedure = args["training_method"]
    ddp_model, optimizer, logs, args = optimization_procedure(
        ddp_model, optimizer, logs, distributed_trainer, dataloader_train, val_dataset, checkpointer, args
    )

    runtime = time.time() - t1

    # Save results
    save_results(distributed_trainer, logs, ddp_model, optimizer, args, other_settings, dataloaders, runtime)

    if cleanup:
        distributed_trainer.cleanup()

def set_seed(seed: int):
    """
    Set random seeds for reproducibility across all random number generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#### Helper functions ####

def create_dataloader(dataset: ByteWikipediaDataset, world_size: int, rank: int, args: dict) -> DataLoader:
    """
    Create dataloader with appropriate batch size for distributed training.
    """
    bs = args["batch_size"]
    assert bs % world_size == 0, "Batch size must be divisible by the number of GPUs."
    assert bs >= world_size, "Batch size must be larger than the number of GPUs. Otherwise, on-line learning is not realizable with DDP."

    dataloader = DataLoader(dataset, batch_size=bs, shuffle = False)
    return dataloader

def save_results(distributed_trainer: DistributedTransformerTrainer, logs: dict, ddp_model: DDP, optimizer, args: dict, other_settings: dict, dataloaders: list, runtime: float):
    """
    Save training results, model weights, and metrics to disk.
    
    Only primary process (rank 0) performs the actual saving.
    """
    run, run_df, rank = distributed_trainer.run, distributed_trainer.run_df, distributed_trainer.rank

    model = ddp_model.module

    test_loss, train_loss, model_size, non_zero_params, on_line_code_length = calculate_some_metrics(distributed_trainer, ddp_model, dataloaders, args, other_settings, logs)

    if rank != 0: # Only rank 0 should save the results to file
        return

    distributed_trainer.save_csv(logs["train_loss_X"], logs["train_loss"], "train_loss")
    distributed_trainer.save_csv(logs["runtime_per_250_batches"], logs["runtime_per_250_batches"], "runtime_per_250_batches")
    distributed_trainer.save_csv(logs["l0_norm_X"], logs["l0_norm"], "l0_norm")

    torch.save(model.state_dict(), os.path.join(run.path, "model.pth"))
    torch.save(optimizer.state_dict(), os.path.join(run.path, "optimizer.pth"))

    run_df.log_meta()
    run_df.log_args(args)

    # log metrics that are based on the logs dictionary
    run_df.log_metric("final_train_loss", logs["train_loss"][-1])
    run_df.log_metric("training_runtime", runtime)
    if train_loss:
        run_df.log_metric("mean_train_loss", train_loss)
    if test_loss:
        run_df.log_metric("mean_test_loss", test_loss)
    if model_size:
        run_df.log_metric("model_byte_size", model_size)
    if non_zero_params:
        run_df.log_metric("non_zero_params", non_zero_params)
    if on_line_code_length:
        run_df.log_metric("description_length", on_line_code_length)

    run_df.save_to_CSV()

    return

def calculate_some_metrics(distributed_trainer, ddp_model, dataloaders, args, other_settings, logs):
    """
    Calculate various model performance and size metrics based on settings.
    
    Metrics include loss values, model size, parameter counts, and code length.
    """
    dataloader_train, dataloader_val, dataloader_test = dataloaders

    rank = distributed_trainer.rank
    device = distributed_trainer.device
    model = ddp_model.module

    if other_settings["calculate_train_loss"]:
        print("Calculating train loss")
        train_loss = loss_over_dataset(ddp_model, dataloader_train, args, distributed_trainer, debug=other_settings["debug"])
        print("Train loss: ", train_loss)
    else:
        train_loss = None

    if other_settings["calculate_test_loss"]:
        print("Calculating test loss")
        test_loss = loss_over_dataset(ddp_model, dataloader_test, args, distributed_trainer, debug=other_settings["debug"])
        print("Test loss: ", test_loss)
    else:
        test_loss = None

    if other_settings["calculate_model_byte_size"] and rank == 0:
        print("Calculating model size")
        model_size = ModelByteSize.byte_size(model)
        print("Model size: ", model_size)
    else:
        model_size = None

    if other_settings["calculate_non_zero_params"] and rank == 0:
        print("Calculating non-zero parameters")
        non_zero_params = ModelByteSize.get_model_info(model, device)["non_zero_params"]
        print("Non-zero parameters: ", non_zero_params)
    else:
        non_zero_params = None
    
    if other_settings["calculate_on_line_code_length"] and rank == 0:
        print("Calculating on-line code length")
        code_length = online_code_length_from_dict(logs, args)
        print("On-line code length: ", code_length)
    else:
        code_length = None

    return test_loss, train_loss, model_size, non_zero_params, code_length