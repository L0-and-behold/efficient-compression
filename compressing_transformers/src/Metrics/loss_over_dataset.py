import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from src.DistributedTransformerTrainer import DistributedTransformerTrainer
from tqdm.auto import tqdm
import math

def loss_over_dataset(ddp_model: DDP, dataloader: DataLoader, args: dict, distributed_trainer: DistributedTransformerTrainer, debug=False, loss_type="") -> float:
    """Calculate average loss over an entire dataset using distributed evaluation.
    
    Efficiently computes loss by splitting batches across available GPUs and
    aggregating results. Supports progress tracking and debug mode for quick testing.
    
    Args:
        ddp_model (DDP): Distributed model to evaluate
        dataloader (DataLoader): Dataset loader containing batches to evaluate
        args (dict): Configuration parameters including batch size
        distributed_trainer (DistributedTransformerTrainer): Trainer with distribution info
        debug (bool, optional): Run only one batch if True. Defaults to False.
        loss_type (str, optional): Can take values "train", "test" or "". Defaults to "". If set to train, then args["only_process_every_nth_batch_when_calculating_train_loss"] is used to determine which fraction of batches is used for training. If set to "train", then args["only_process_every_nth_batch_when_calculating_test_loss"] is used instead. And if set to "" (or any other string), then every batch of the dataset is processed by default.
    
    Returns:
        float: Average loss value across all evaluated batches
    """
    world_size, rank, device = distributed_trainer.world_size, distributed_trainer.rank, distributed_trainer.device
    
    ddp_model.eval()
    total_loss = 0
    evaluated_batches = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    if loss_type == "train":
        only_process_every_nth_batch = args["only_process_every_nth_batch_when_calculating_train_loss"]
    elif loss_type == "test":
        only_process_every_nth_batch = args["only_process_every_nth_batch_when_calculating_test_loss"]
    else:
        only_process_every_nth_batch = 1

    # Split the batch across GPUs
    local_batch_size = args["batch_size"] // world_size
    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size

    with torch.no_grad():
        if rank == 0:
            progress_bar = tqdm(total=100, desc=f"Evaluating", disable=not rank == 0)
            update_interval = max(1, len(dataloader) // 100)
            
        for (i, batch) in enumerate(dataloader):
            if i % only_process_every_nth_batch != 0:
                continue
            # Split the batch across GPUs
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