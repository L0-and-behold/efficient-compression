import time
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Any, Union
from src.CheckpointHandler import CheckpointHandler

from src.DistributedOptimizationProcedures.BaseProcedure import ProcedureTrainer, get_optimization_procedure

class L1Trainer(ProcedureTrainer):
    """
    Trainer implementing L1 regularization for inducing sparsity
    in neural network parameters.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize L1 trainer with the regularization strength alpha.
        """
        super().__init__(*args, **kwargs)

        self.alpha = self.args["alpha"]

    # Overrides the parent class method
    def modify_grads(self):
        """
        Apply L1 regularization to all model parameters.
        """
        for param in self.ddp_model.parameters():
            self.modify_grad(param)

    def modify_grad(self, param):
        """
        Add the subgradient of L1 penalty (sign of parameter) to parameter gradients.
        """
        if torch.is_tensor(param.grad):
            param.grad += self.alpha*torch.sign(param.data)


def rl1_procedure(
        ddp_model: DDP,
        optimizer: Optimizer,
        logs: Dict[str, Any],
        distributed_trainer, 
        dataloader_train: DataLoader,
        val_dataset, 
        checkpointer: CheckpointHandler, 
        args: Dict[str, Any],
        ) -> tuple[DDP, Optimizer, Dict[str, list]]:
    """
    Execute the L1 regularization procedure with pruning and fine-tuning phases.
    Returns the trained model, optimizer, and training logs.
    """
    return get_optimization_procedure(L1Trainer, ddp_model, optimizer, logs, distributed_trainer, dataloader_train, val_dataset, checkpointer, args)

