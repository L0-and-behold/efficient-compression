import time
import copy
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Union
from src.CheckpointHandler import CheckpointHandler
from src.Tamade import Tamade
from src.DistributedTransformerTrainer import DistributedTransformerTrainer
from src.Transformer.Mask import Mask

class ProcedureTrainer:
    """Base class for distributed training procedures with support for pruning,
    fine-tuning, and checkpoint management.
    """

    def __init__(self, ddp_model: DDP, optimizer: Optimizer, device: torch.device, args: Dict[str, Any], world_size: int, rank: int):
        """Initialize the procedure trainer with model, optimizer and distributed training configuration.
        
        Args:
            ddp_model: Distributed model for training
            optimizer: Optimizer for parameter updates
            device: Computation device (CPU/GPU)
            args: Configuration parameters
            world_size: Total number of processes
            rank: Current process rank
        """
        self.ddp_model = ddp_model
        self.optimizer = optimizer
        self.device = device
        self.args = args
        self.world_size = world_size
        self.rank = rank
        self.mask = Mask(ddp_model.module)
        self.current_epoch = 0
        
        assert isinstance(self.args["tolerated_relative_loss_increase"], float), "tolerated_relative_loss_increase must be a float"
        assert isinstance(self.args["do_pruning"], bool), "do_pruning must be a boolean"
        assert isinstance(self.args["alpha"], float), "alpha must be a float"
        assert isinstance(self.args["epochs"], int), "epochs must be an integer"
        assert isinstance(self.args["stop_epoch_at_batch"], (bool, int)), "stop_epoch_at_batch must be a boolean or integer"
        assert isinstance(self.args["steps_per_chunk"], int), "steps_per_chunk must be an integer"
        assert isinstance(self.args["log_every"], int), "log_every must be an integer"

    def process_chunk_fine_tuning(self, batch):
        """Process a batch in fine-tuning mode, disabling regularization modifications.
        
        Args:
            batch: Input data batch
            
        Returns:
            Loss value for the processed batch
        """
        return self.process_chunk(batch, prelude_finetuning=True)

    def process_chunk(self, batch: torch.Tensor, prelude_finetuning=False):
        """Process a single batch for training, applying mask and managing gradients.
        
        Args:
            batch: Input data batch
            prelude_finetuning: Whether to skip regularization modifications
            
        Returns:
            Loss value for the processed batch
        """
        # Split the batch across GPUs
        local_batch_size = self.args["batch_size"] // self.world_size
        start_index = self.rank * local_batch_size
        end_index = start_index + local_batch_size
        local_batch = batch[start_index:end_index].to(self.device)
        input_seq = local_batch[:, :-1]
        target_seq = local_batch[:, 1:]
        
        self.optimizer.zero_grad()
        self.mask(self.ddp_model) # Apply mask to model. By default, this is an identity operation
        output = self.ddp_model(input_seq)
        output = output.contiguous().view(-1, output.size(-1))
        target_seq = target_seq.contiguous().view(-1)
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target_seq)
        loss.backward()
        
        # Modify gradients if applicable
        if not prelude_finetuning:
            self.modify_grads()
            
        self.optimizer.step()
        self.mask(self.ddp_model)
        
        if not prelude_finetuning:
            self.project_params()
            
        return loss

    def modify_grads(self):
        """Hook to modify gradients before optimization step.
        
        To be implemented by subclasses.
        """
        # Override this method in subclasses
        pass

    def project_params(self):
        """Hook to project parameters after optimization step.
        
        To be implemented by subclasses.
        """
        # Override this method in subclasses
        pass

    def pruning_procedure(self, ddp_model, validation_set, args, distributed_trainer: DistributedTransformerTrainer):
        """Execute threshold adaptive mask determination (TAMADE) to find optimal sparsity level.
        
        Args:
            ddp_model: Distributed model to prune
            validation_set: Dataset used for validating pruning impact
            args: Configuration parameters
            distributed_trainer: Trainer for distributed operations
            
        Returns:
            tuple: (pruned_model, updated_args)
        """
        if not args["do_pruning"]:
            return ddp_model, args
            
        print("Starting threshold adaptive mask determination (TAMADE)")
        t = time.time()
        bsp = Tamade(
            self.device, self.world_size, self.rank, args["batch_size"],
            validation_set,
            tolerated_relative_loss_increase=args["tolerated_relative_loss_increase"],
            epsilon_min=0.0, epsilon_max=0.8, tol=1e-7
        )
        pruned_model, epsilon, steps = bsp(ddp_model, distributed_trainer)
        
        if self.rank == 0:
            print("Binary search pruning finished in ", time.time()-t, " seconds using ", steps, " steps")
            
        args["epsilon"] = epsilon
        self.mask.update(pruned_model.module, epsilon) # Update mask with new epsilon
        return pruned_model, args

    def train_epoch(self, dataloader: DataLoader, checkpointer: CheckpointHandler, logs: dict, rank: int, break_at_batch: Union[bool, int], prelude_finetuning: bool = False) -> bool:
        """Train for a single epoch with support for checkpointing and early stopping.
        
        Args:
            dataloader: Data provider for training
            checkpointer: Handler for saving model state
            logs: Dictionary for tracking metrics
            rank: Current process rank
            break_at_batch: If int, stop epoch at specified batch; if False, run full epoch
            prelude_finetuning: Whether to run in fine-tuning mode
            
        Returns:
            bool: True if training should continue, False otherwise
        """
        epochs = self.args["epochs"]
        log_every = self.args["log_every"]
        steps_per_chunk = self.args["steps_per_chunk"]
        self.ddp_model.train()
        t = time.time()
        
        # Progress bar
        if rank == 0:
            print(f"Training epoch {checkpointer.epoch+1}/{epochs}")
            pbar = tqdm(total=len(dataloader), desc=f"Epoch {checkpointer.epoch+1}/{epochs}", position=0, leave=True)
            if checkpointer.chunk > 0:
                pbar.update(checkpointer.chunk)
            else:
                pbar = None
        
        # Training for one epoch
        iter_runtime = time.time()
        for i, batch in enumerate(dataloader):
            if i < checkpointer.chunk:
                continue
                
            for _ in range(steps_per_chunk):
                # Training Step
                loss = self.process_chunk(batch, prelude_finetuning)
                
                # Logging
                self._logging(rank, pbar, log_every, logs, loss, iter_runtime, i, self.current_epoch)
                
                # Checkpointing / Early stopping
                break_now = self._checkpointing_early_stopping(break_at_batch, i, checkpointer, logs, rank, pbar)
                if break_now:
                    return True
                    
        return True

    def _logging(self, rank: int, pbar, log_every: int, logs: dict, loss: torch.Tensor, iter_runtime: float, i: int, epoch: int):
        """Handle logging of training metrics.
        
        Args:
            rank: Current process rank
            pbar: Progress bar instance
            log_every: Frequency of logging
            logs: Dictionary for tracking metrics
            loss: Current loss value
            iter_runtime: Timestamp for runtime calculation
            i: Current batch index
            epoch: Current epoch
        """
        if rank == 0:
            pbar.update(1)
            if (i + 1) % log_every == 0:
                logs["train_loss"].append(loss.item())
                logs["train_loss_X"].append(i + 1)
                logs["train_loss_X_epoch"].append(epoch)
                
            if (i + 1) % (log_every * 10) == 0:
                logs["l0_norm"].append(len(self.mask))
                logs["l0_norm_X"].append(i + 1)
                logs["l0_norm_X_epoch"].append(epoch)
                
            if i % 100 == 0:
                logs["runtime_per_250_batches"].append(time.time() - iter_runtime)
                iter_runtime = time.time()
                
            if (i + 1) % log_every == 0:
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

    def _checkpointing_early_stopping(self, break_at_batch, i: int, checkpointer: CheckpointHandler, logs: dict, rank: int, pbar):
        """Manage checkpointing and early stopping conditions.
        
        Args:
            break_at_batch: Batch index to stop at, or False to run full epoch
            i: Current batch index
            checkpointer: Handler for saving model state
            logs: Dictionary of training metrics
            rank: Current process rank
            pbar: Progress bar instance
            
        Returns:
            bool: True if training should be stopped, False otherwise
        """
        break_here = False
        if break_at_batch and i >= break_at_batch:
            print(f"Stop epoch {checkpointer.epoch+1} early after chunk {i}.")
            break_here = True
            
        if checkpointer.should_checkpoint():
            if rank == 0:
                checkpointer.update(checkpointer.epoch, i + 1)
                checkpointer.save_checkpoint(self.ddp_model, self.optimizer, logs)
                print(f"Checkpoint saved at epoch {checkpointer.epoch+1}, chunk {i+1}")
            break_here = True
            
        if break_here and rank == 0:
            pbar.close()
            
        return break_here

def get_optimization_procedure(
    trainer_class, # such as VanillaTrainer or DrrTrainer
    ddp_model: DDP,
    optimizer: Optimizer,
    logs: Dict[str, Any],
    distributed_trainer: DistributedTransformerTrainer,
    dataloader_train: DataLoader,
    validation_set,
    checkpointer: CheckpointHandler,
    args: Dict[str, Any],
) -> callable:
    """Main training workflow: executes prelude training, main training loop with pruning,
    and final fine-tuning phase using the specified trainer class.
    
    Args:
        trainer_class: Class to instantiate for training (e.g., VanillaTrainer, DrrTrainer)
        ddp_model: Distributed model for training
        optimizer: Optimizer for parameter updates
        logs: Dictionary for tracking metrics
        distributed_trainer: Trainer managing distributed operations
        dataloader_train: Data provider for training
        validation_set: Dataset for validation during pruning
        checkpointer: Handler for saving model state
        args: Configuration parameters
        
    Returns:
        tuple: (trained_model, optimizer, logs, updated_args)
    """
    assert isinstance(args["epochs"], int), "epochs must be an integer"
    assert isinstance(args["stop_epoch_at_batch"], (bool, int)), "stop_epoch_at_batch must be a boolean or integer"
    assert isinstance(args["epochs_fine_tuning"], int), "epochs must be an integer"
    assert isinstance(args["stop_epoch_at_batch_fine_tuning"], (bool, int)), "stop_epoch_at_batch must be a boolean or integer"
    
    world_size, rank = distributed_trainer.world_size, distributed_trainer.rank
    trainer = trainer_class(ddp_model, optimizer, distributed_trainer.device, args, world_size, rank)
    
    # 1. Prelude training loop (training without regularization before main training loop)
    epochs = args["epochs_prelude"]
    break_at_batch = args["stop_epoch_at_batch_prelude"]
    if epochs > 0 and rank == 0:
        print("Starting prelude training loop")
        
    for _ in range(epochs):
        # Training loop
        continue_training = trainer.train_epoch(dataloader_train, checkpointer, logs, rank, break_at_batch, prelude_finetuning=True)
        if not continue_training:
            raise("Exiting prelude training loop since max runtime reached or early stopping triggered. Checkpointing not implemented for prelude training or fine-tuning.")
    
    # 2. Main training loop
    epochs = args["epochs"]
    break_at_batch = args["stop_epoch_at_batch"]
    if epochs > 0 and rank == 0:
        print("Starting main training loop")
        
    for epoch in range(checkpointer.epoch, epochs):
        trainer.current_epoch = epoch
        
        # Train for a single epoch
        continue_training = trainer.train_epoch(dataloader_train, checkpointer, logs, rank, break_at_batch, prelude_finetuning=False)
        
        # Pruning procedure
        if (epoch + 1) >= args["first_pruning_after"] and (epoch + 1) % args["prune_every"] == 0:
            pruned_model, args = trainer.pruning_procedure(ddp_model, validation_set, args, distributed_trainer)
            ddp_model.module.load_state_dict(pruned_model.module.state_dict())
            trainer.ddp_model = ddp_model
            
        if not continue_training:
            print("Exiting epoch loop since max runtime reached or early stopping triggered")
            break
            
        checkpointer.epoch += 1
    
    # 3. Fine-tuning loop
    epochs = args["epochs_fine_tuning"]
    break_at_batch = args["stop_epoch_at_batch_fine_tuning"]
    if epochs > 0 and rank == 0:
        print("Starting fine-tuning loop")
        
    for _ in range(epochs):
        # Training loop
        continue_training = trainer.train_epoch(dataloader_train, checkpointer, logs, rank, break_at_batch, prelude_finetuning=True)
        if not continue_training:
            raise("Exiting fine-tuning loop since max runtime reached or early stopping triggered. Checkpointing not implemented for fine-tuning.")
    
    # since the fine tuning epochs and regular epochs are not differentiated in the logs, we opt to use the last checkpoint before fine-tuning
    return ddp_model, optimizer, logs, args