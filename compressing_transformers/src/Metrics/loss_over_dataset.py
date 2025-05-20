import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from src.DistributedTransformerTrainer import DistributedTransformerTrainer
from tqdm.auto import tqdm
import math

def loss_over_dataset(ddp_model: DDP, dataloader: DataLoader, args: dict, distributed_trainer: DistributedTransformerTrainer, debug=False) -> float:
    """Calculate average loss over an entire dataset using distributed evaluation.
    
    Efficiently computes loss by splitting batches across available GPUs and
    aggregating results. Supports progress tracking and debug mode for quick testing.
    
    Args:
        ddp_model (DDP): Distributed model to evaluate
        dataloader (DataLoader): Dataset loader containing batches to evaluate
        args (dict): Configuration parameters including batch size
        distributed_trainer (DistributedTransformerTrainer): Trainer with distribution info
        debug (bool, optional): Run only one batch if True. Defaults to False.
    
    Returns:
        float: Average loss value across all evaluated batches
    """
    world_size, rank, device = distributed_trainer.world_size, distributed_trainer.rank, distributed_trainer.device
    
    ddp_model.eval()
    total_loss = 0
    evaluated_batches = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        if rank == 0:
            progress_bar = tqdm(total=100, desc=f"Evaluating", disable=not rank == 0)
            update_interval = max(1, len(dataloader) // 100)
            
        for (i, batch) in enumerate(dataloader):
            # Split the batch across GPUs
            local_batch_size = args["batch_size"] // world_size
            start_index = rank * local_batch_size
            end_index = start_index + local_batch_size
            local_batch = batch[start_index:end_index].to(device)
            
            input_seq = local_batch[:, :-1]
            target_seq = local_batch[:, 1:]
            
            output = ddp_model(input_seq)
            output = output.contiguous().view(-1, output.size(-1))
            target_seq = target_seq.contiguous().view(-1)
            
            loss = criterion(output, target_seq)
            total_loss += loss.item()
            evaluated_batches += 1
            
            if rank == 0 and (i % update_interval == 0 or i == len(dataloader) - 1):
                progress = math.floor((i + 1) / len(dataloader) * 100)
                progress_bar.update(progress - progress_bar.n)
                progress_bar.set_postfix({"loss": total_loss / evaluated_batches})
                
            if debug:
                break
                
        if rank == 0:
            progress_bar.close()
            
    return total_loss / evaluated_batches