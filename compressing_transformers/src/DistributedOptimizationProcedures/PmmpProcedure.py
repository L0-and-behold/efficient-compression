import time
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Any, Union
from src.CheckpointHandler import CheckpointHandler
from src.DistributedOptimizationProcedures.BaseProcedure import ProcedureTrainer, get_optimization_procedure

class PmmpTrainer(ProcedureTrainer):
    """Trainer implementing Probabilistic Minimax Pruning (PMMP) strategy for efficient neural network compression.
    
    PMMP uses a probabilistic approach to prune parameters through minimax optimization.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize PMMP trainer with regularization parameter and compute the total number of trainable parameters.
        
        Args:
            *args: Positional arguments passed to parent class
            **kwargs: Keyword arguments passed to parent class
        """
        super().__init__(*args, **kwargs)
        self.alpha = self.args["alpha"]
        self.model_param_number = sum(p.numel() for p in self.ddp_model.parameters() if p.requires_grad)

    # Overrides the parent class method
    def modify_grads(self):
        """Apply PMMP gradient modifications to all parameter quadruplets.
        
        Processes all (a, w, p, u) parameter groups in the model.
        """
        m = self.ddp_model
        for (a, w, p, u) in zip(m.parameters(), m.module.parameters_w(), m.module.parameters_p(), m.module.parameters_u()):
            self.modify_grad(a, w, p, u)
            
    def modify_grad(self, a, w, p, u):
        """Apply PMMP gradient modifications to a parameter quadruplet.
        
        Implements the gradient modifications for the minimax optimization strategy
        by updating gradients for each parameter type.
        
        Args:
            a: Actual parameter
            w: Weight parameter
            p: Probability parameter
            u: Lagrangian multiplier
        """
        if torch.is_tensor(a.grad):
            wp = w * p
            awp = a - wp
            diff = u * 2 * awp
            wp1p = wp * (1 - p)
            a.grad += diff
            w.grad = - diff * p + u * 2 * wp1p
            p.grad = - diff * w + u * w**2 * (1 - 2 * p) + self.alpha
            u.grad = - (awp**2 + w * wp1p) 
            # this division by self.model_param_number is only recommended if u is a tensor 
            # with a single entry and not if it is a tensor of the shape of p

    def project_params(self):
        """Apply PMMP parameter projections after gradient updates.
        
        Ensures probability parameters remain in valid range [0,1]
        to maintain the probabilistic interpretation.
        """
        with torch.no_grad():
            for p in self.ddp_model.module.parameters_p():
                if p.requires_grad:
                    p.clamp_(min=0, max=1)

def pmmp_procedure(
    ddp_model: DDP,
    optimizer: Optimizer,
    logs: Dict[str, Any],
    distributed_trainer,
    dataloader_train: DataLoader,
    val_dataset,
    checkpointer: CheckpointHandler,
    args: Dict[str, Any],
):
    """Execute the Probabilistic Minimax Pruning optimization procedure.
    
    Implements the complete PMMP training workflow with phases for pruning
    and fine-tuning to produce a compressed model.
    
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
    return get_optimization_procedure(PmmpTrainer, ddp_model, optimizer, logs, distributed_trainer, dataloader_train, val_dataset, checkpointer, args)