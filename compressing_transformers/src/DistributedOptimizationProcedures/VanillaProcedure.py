from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Any, Union
from src.CheckpointHandler import CheckpointHandler
from src.DistributedOptimizationProcedures.BaseProcedure import ProcedureTrainer, get_optimization_procedure

class VanillaTrainer(ProcedureTrainer):
    """Standard training procedure implementation without any regularization or specialized optimization techniques.
    
    Provides a baseline implementation that follows the standard training workflow
    without additional complexity, useful for establishing performance benchmarks.
    """
    
    # as Base, but no pruning
    def pruning_procedure(self, ddp_model, validation_set, args, distributed_trainer):
        """Override base pruning procedure to skip pruning entirely.
        
        Args:
            ddp_model: Distributed model that would normally be pruned
            validation_set: Dataset used for validation (unused in this implementation)
            args: Configuration parameters
            distributed_trainer: Trainer for distributed operations (unused in this implementation)
            
        Returns:
            tuple: (unchanged_model, unchanged_args)
        """
        return ddp_model, args

def vanilla_procedure(
    ddp_model: DDP,
    optimizer: Optimizer,
    logs: Dict[str, Any],
    distributed_trainer,
    dataloader_train: DataLoader,
    validation_set,
    checkpointer: CheckpointHandler,
    args: Dict[str, Any]
):
    """Execute the standard training procedure without specialized regularization.
    
    Implements the basic training workflow as a control/baseline without
    additional regularization or optimization techniques.
    
    Args:
        ddp_model: Distributed model to train
        optimizer: Optimizer for parameter updates
        logs: Dictionary for tracking metrics
        distributed_trainer: Trainer managing distributed operations
        dataloader_train: Data provider for training
        validation_set: Dataset for validation
        checkpointer: Handler for saving model state
        args: Configuration parameters
    
    Returns:
        (trained_model, optimizer, logs, updated_args)
    """
    return get_optimization_procedure(VanillaTrainer, ddp_model, optimizer, logs, distributed_trainer, dataloader_train, validation_set, checkpointer, args)