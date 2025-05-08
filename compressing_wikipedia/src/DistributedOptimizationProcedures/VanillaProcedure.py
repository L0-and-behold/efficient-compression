from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Any, Union
from src.CheckpointHandler import CheckpointHandler

from src. DistributedOptimizationProcedures.BaseProcedure import ProcedureTrainer, get_optimization_procedure

class VanillaTrainer(ProcedureTrainer):
    """
    Standard training procedure implementation without any regularization 
    or specialized optimization techniques.
    """
    # as Base, but no pruning
    def pruning_procedure(self, ddp_model, validation_set, args, distributed_trainer):
        """
        Override base pruning procedure to skip pruning entirely.
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
        ) -> tuple[DDP, Optimizer, Dict[str, list]]:
    """
    Execute the standard training procedure without specialized regularization.
    Returns the trained model, optimizer, and training logs.
    """
    return get_optimization_procedure(VanillaTrainer, ddp_model, optimizer, logs, distributed_trainer, dataloader_train, validation_set, checkpointer, args)

