import time
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Any, Union
from src.CheckpointHandler import CheckpointHandler
from src.DistributedOptimizationProcedures.BaseProcedure import ProcedureTrainer, get_optimization_procedure

class L1Trainer(ProcedureTrainer):
    """Trainer implementing L1 regularization for inducing sparsity in neural network parameters.
    
    L1 regularization adds a penalty term proportional to the absolute value of weights,
    encouraging many parameters to become exactly zero during training.
    This can also be interpreted as a smooth approximation of ℓ₀-regularization.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize L1 trainer with the regularization strength alpha.
        
        Args:
            *args: Positional arguments passed to parent class
            **kwargs: Keyword arguments passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.alpha = self.args["alpha"]

    # Overrides the parent class method
    def modify_grads(self):
        """Apply L1 regularization to all model parameters.
        
        Iterates through all parameters and applies the L1 gradient modification.
        """
        for param in self.ddp_model.parameters():
            self.modify_grad(param)
            
    def modify_grad(self, param):
        """Add the subgradient of L1 penalty (sign of parameter) to parameter gradients.
        
        This implements the core L1 regularization logic by adding the sign of each
        parameter multiplied by the regularization strength to the gradients.
        
        Args:
            param: Parameter to apply L1 regularization to
        """
        if torch.is_tensor(param.grad):
            param.grad += self.alpha * torch.sign(param.data)

def rl1_procedure(
    ddp_model: DDP,
    optimizer: Optimizer,
    logs: Dict[str, Any],
    distributed_trainer,
    dataloader_train: DataLoader,
    val_dataset,
    checkpointer: CheckpointHandler,
    args: Dict[str, Any],
):
    """Execute the L1 regularization procedure with pruning and fine-tuning phases.
    
    Implements a complete training workflow using L1 regularization to induce
    sparsity, followed by pruning and fine-tuning steps.
    
    Args:
        ddp_model: Distributed model to train
        optimizer: Optimizer for parameter updates
        logs: Dictionary for tracking metrics
        distributed_trainer: Trainer managing distributed operations
        dataloader_train: Data provider for training
        val_dataset: Dataset for validation during pruning
        checkpointer: Handler for saving model state
        args: Configuration parameters
    
    Returns:
        (trained_model, optimizer, logs, updated_args)
    """
    return get_optimization_procedure(L1Trainer, ddp_model, optimizer, logs, distributed_trainer, dataloader_train, val_dataset, checkpointer, args)