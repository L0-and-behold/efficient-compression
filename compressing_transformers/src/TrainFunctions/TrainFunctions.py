import os
import time
import torch
import csv
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
from src.DistributedOptimizationProcedures.PmmpProcedure import pmmp_procedure


class TrainFunctions:
    """Contains the logic of the different functions that need to be called during training.
    """
    
    def __init__(self, args: dict, other_settings: dict, path_to_database: str, experiment_name: str):
        """Initialize the train logic.
        
            Sets up the distributed environment, initializes the transformer model and trainer

        Args:
            args: Dictionary containing all training and model configuration parameters.
            other_settings: Dictionary with flags for metrics calculation.
            path_to_database: Directory path where results will be stored.
            experiment_name: Name of this experiment for organization and logging.
        """
        self.args = args
        self.other_settings = other_settings

        # Set the multiprocessing start method to 'spawn' for CUDA compatibility
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        
        # Set random seeds for reproducibility
        self.set_seed(self.args["seed"])

        # Set PMMP flag
        if self.args["training_procedure"] == pmmp_procedure:
            self.args["pmmp"] = True
        else:
            self.args["pmmp"] = False

        # Create results directory if it doesn't exist
        os.makedirs(path_to_database, exist_ok=True)

        # Load transformer configuration based on the selected model size
        transformer_config = TransformerConfig(self.args["transformer_config"])()
        transformer_config["learning_rate"] = self.args["learning_rate"]  # Override default learning rate
        # Initialize the distributed trainer and checkpoint handler
        self.distributed_trainer = DistributedTransformerTrainer(self.args, path_to_database, experiment_name, transformer_config, seed=self.args["seed"])
        self.checkpointer = CheckpointHandler(experiment_name, self.args["checkpoint_time"], self.args["max_runtime"])

        self.args["total_number_of_GPUs"] = self.distributed_trainer.world_size
        self.args["batch_size_per_gpu"] = transformer_config["batch_size_per_gpu"] # storing this in args for logging purposes

        self.args["seq_length"] = transformer_config["seq_length"]  # Store sequence length (=context window) for dataset preparation
        self.args["batch_size"]  = transformer_config["batch_size_per_gpu"]*self.args["total_number_of_GPUs"]
        if self.args["tokens_per_epoch"]:
            self.args["iterations_per_epoch"] = int(self.args["tokens_per_epoch"] / self.args["batch_size"] / self.args["seq_length"])
        self.args["train_only_on_leading_tokens"] = int(self.args["iterations_per_epoch"]*self.args["batch_size"]*self.args["seq_length"]) # Limit training to first N tokens (False or int), here specified in terms of iterations, batch_size and seq_length (context window)

        if self.args["stop_epoch_at_batch_prelude"]:
            prelude_iterations_per_epoch = self.args["stop_epoch_at_batch_prelude"]
        else:
            prelude_iterations_per_epoch = self.args["iterations_per_epoch"]
        if self.args["stop_epoch_at_batch"]:
            main_iterations_per_epoch = self.args["stop_epoch_at_batch"]
        else:
            main_iterations_per_epoch = self.args["iterations_per_epoch"]
        if self.args["stop_epoch_at_batch_fine_tuning"]:
            finetuning_iterations_per_epoch = self.args["stop_epoch_at_batch_fine_tuning"]
        else:
            finetuning_iterations_per_epoch = self.args["iterations_per_epoch"]
        self.args["total_number_of_iterations"] = prelude_iterations_per_epoch * self.args["epochs_prelude"] + main_iterations_per_epoch * self.args["epochs"] + finetuning_iterations_per_epoch * self.args["epochs_fine_tuning"]

        # Verify batch size is compatible with distributed training
        assert self.args["batch_size"] % self.distributed_trainer.world_size == 0, "Batch size must be divisible by the number of GPUs."
        assert self.args["batch_size"] >= self.distributed_trainer.world_size, "Batch size must be larger than the number of GPUs. Otherwise, on-line learning is not realizable with DDP."

        # If training on a subset, verify the subset size is compatible with sequence length and batch size
        if self.args["train_only_on_leading_tokens"]:
            assert self.args["train_only_on_leading_tokens"] % self.args["seq_length"] == 0, f"'train_only_on_leading_tokens' must be a multiple of the sequence length {self.args['seq_length']}"
            assert self.args["train_only_on_leading_tokens"] % self.args["batch_size"] == 0, f"'train_only_on_leading_tokens' must be a multiple of the batch size {self.args['batch_size']}"
        

    def train_and_save_results(self, cleanup=True):
        """Execute the complete training workflow including dataset loading, model training, and result saving.
        
        This function handles the entire training pipeline:
        1. Load and prepare datasets
        2. Initialize or restore model and optimizer
        3. Execute the selected training procedure
        4. Calculate metrics and save results
        
        Args:
            cleanup: Whether to clean up distributed processes after training. Defaults to True.
        """
        # Get process rank and world size for distributed operations
        rank = self.distributed_trainer.rank
        world_size = self.distributed_trainer.world_size
        
        # Load Wikipedia dataset with the configured sequence length
        dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_local_path = os.path.join(dirpath, 'Datasets/processed_wiki_dataset_' + str(self.args["seq_length"]) + '.pt')

        datasets = WikipediaDatasets.load_dataset(dataset_local_path, self.args["seq_length"], rank = rank)

        train_dataset, val_dataset, test_dataset = datasets.train, datasets.validation, datasets.test
        
        # If specified, use only a subset of the training data
        if self.args["train_only_on_leading_tokens"]:  # overwrite trainset with smaller subset
            train_dataset = WikipediaDatasets.get_subset(0, self.args["train_only_on_leading_tokens"], train_dataset)
        
        # Create data loaders for each dataset
        dataloaders = [self.create_dataloader(dataset, world_size) for dataset in (train_dataset, val_dataset, test_dataset)]
        dataloader_train, dataloader_val, dataloader_test = dataloaders
        
        # Initialize model and optimizer or load from checkpoint
        model, ddp_model, optimizer, scheduler = self.distributed_trainer.model_optimizer(warmup_steps=self.args["warmup_steps"], weight_decay=self.args["weight_decay"])
        model_state_dict, optimizer_state_dict, scheduler_state_dict, logs = self.checkpointer.load_checkpoint(rank=rank)
        
        if model_state_dict and optimizer_state_dict and scheduler_state_dict:
            # Restore from checkpoint if available
            ddp_model.module.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optimizer_state_dict)
            scheduler.load_state_dict(scheduler_state_dict)
        
        # Print model information (only from rank 0 process)
        if rank == 0:
            self.distributed_trainer.print_model_info(model)
        
        t1 = time.time()
        
        # Execute the selected training procedure
        optimization_procedure = self.args["training_procedure"]
        new_ddp_model, optimizer, scheduler, logs, self.args = optimization_procedure(
            ddp_model, optimizer, scheduler, logs, self.distributed_trainer, dataloader_train, val_dataset, self.checkpointer, self.args
        )
        
        # Ensure model state is correctly transferred after training
        new_ddp_model.module.load_state_dict(ddp_model.module.state_dict())
        runtime = time.time() - t1
        
        # Calculate metrics and save all results
        self.save_results(logs, ddp_model, optimizer, scheduler, dataloaders, runtime)
        
        # Clean up distributed processes if requested
        if cleanup:
            self.distributed_trainer.cleanup()


    def set_seed(self, seed: int):
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

    def create_dataloader(self, dataset: ByteWikipediaDataset, world_size: int) -> DataLoader:
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
        bs = self.args["batch_size"]
        assert bs % world_size == 0, "Batch size must be divisible by the number of GPUs."
        assert bs >= world_size, "Batch size must be larger than the number of GPUs. Otherwise, on-line learning is not realizable with DDP."
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)
        return dataloader


    def save_results(self, logs: dict, ddp_model: DDP, optimizer, scheduler, dataloaders: list, runtime: float):
        """Save training results, model weights, and metrics to disk.
        
        This function handles the saving of:
        1. Log data as CSV files
        2. Model and optimizer and scheduler states
        3. Metadata and arguments
        4. Calculated metrics
        
        Only the primary process (rank 0) performs the actual saving.
        
        Args:
            logs: Dictionary containing training logs and metrics history.
            ddp_model: The trained DistributedDataParallel model.
            optimizer: The optimizer used for training.
            scheduler: The scheduler used for training.
            dataloaders: List of dataloaders for training, validation, and testing.
            runtime: Total training runtime in seconds.
        """
        run, run_df, rank = self.distributed_trainer.run, self.distributed_trainer.run_df, self.distributed_trainer.rank
        model = ddp_model.module
        
        # Calculate various metrics based on settings
        test_loss, train_loss, model_size, non_zero_params, on_line_code_length = self.calculate_some_metrics(ddp_model, dataloaders, logs)
        
        # Only rank 0 should save the results to file
        if rank != 0:
            return
        
        # Save training logs as CSV files
        self.distributed_trainer.save_csv(logs["train_loss_X"], logs["train_loss"], "train_loss")
        self.distributed_trainer.save_csv(logs["runtime_per_250_batches"], logs["runtime_per_250_batches"], "runtime_per_250_batches")
        self.distributed_trainer.save_csv(logs["l0_norm_X"], logs["l0_norm"], "l0_norm")
        
        # Save model and optimizer and scheduler states
        torch.save(model.state_dict(), os.path.join(run.path, "model.pth"))
        torch.save(optimizer.state_dict(), os.path.join(run.path, "optimizer.pth"))
        torch.save(scheduler.state_dict(), os.path.join(run.path, "scheduler.pth"))
        
        # Save args
        with open(os.path.join(run.path, "args.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            for k, v in self.args.items():
                writer.writerow([k, v])

        # Log metadata and configuration parameters
        run_df.log_meta()
        run_df.log_args(self.args)
        
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


    def calculate_some_metrics(self, ddp_model: DDP, dataloaders: list, logs: dict):
        """Calculate various model performance and size metrics based on settings.
        
        Selectively calculates metrics based on the flags in self.other_settings:
        1. Training and test loss
        2. Model size in bytes
        3. Count of non-zero parameters
        4. Online code length
        
        Args:
            ddp_model: The trained DistributedDataParallel model.
            dataloaders: List of dataloaders for train, validation, and test datasets.
            logs: Dictionary containing training logs.
            
        Returns:
            tuple: Contains (test_loss, train_loss, model_size, non_zero_params, code_length),
                with None for metrics that were not calculated.
        """
        dataloader_train, dataloader_val, dataloader_test = dataloaders
        rank = self.distributed_trainer.rank
        device = self.distributed_trainer.device
        model = ddp_model.module
        
        # Calculate training loss if requested
        if self.other_settings["calculate_train_loss"]:
            if rank == 0:
                print("Calculating train loss")
            train_loss = loss_over_dataset(ddp_model, dataloader_train, self.args, self.distributed_trainer, debug=self.other_settings["debug"], only_process_every_nth_batch=self.args["only_process_every_nth_batch_when_calculating_train_loss"])
            if rank == 0:
                print("Train loss: ", train_loss)
        else:
            train_loss = None
        
        # Calculate test loss if requested
        if self.other_settings["calculate_test_loss"]:
            if rank == 0:
                print("Calculating test loss")
            test_loss = loss_over_dataset(ddp_model, dataloader_test, self.args, self.distributed_trainer, debug=self.other_settings["debug"], only_process_every_nth_batch=self.args["only_process_every_nth_batch_when_calculating_test_loss"])
            if rank == 0:
                print("Test loss: ", test_loss)
        else:
            test_loss = None
        
        # Calculate model size in bytes if requested (rank 0 only)
        if self.other_settings["calculate_model_byte_size"] and rank == 0:
            print("Calculating model size")
            model_size = ModelByteSize.byte_size(model)
            print("Model size: ", model_size)
        else:
            model_size = None
        
        # Count non-zero parameters if requested (rank 0 only)
        if self.other_settings["calculate_non_zero_params"] and rank == 0:
            print("Calculating non-zero parameters")
            non_zero_params = ModelByteSize.get_model_info(model, device)["non_zero_params"]
            print("Non-zero parameters: ", non_zero_params)
        else:
            non_zero_params = None
        
        # Calculate online code length if requested (rank 0 only)
        if self.other_settings["calculate_on_line_code_length"] and rank == 0:
            print("Calculating on-line code length")
            code_length = online_code_length_from_dict(logs, self.args)
            print("On-line code length: ", code_length)
        else:
            code_length = None
        
        return test_loss, train_loss, model_size, non_zero_params, code_length