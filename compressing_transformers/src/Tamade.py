import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from src.DistributedTransformerTrainer import DistributedTransformerTrainer
from src.Transformer.TransformerDecoder import MultiLayerTransformerDecoder
import copy

class Tamade:
    """Performs threshold adaptive mask determination (TAMADE) on a DDP model.
    
    TAMADE assumes that many parameters of a trained model are small and do not 
    contribute much to the computed function. Such a model, subject to global 
    magnitude pruning (GMP) with a threshold epsilon > 0 is such, that:
    
    1) The higher epsilon, the more parameters are being pruned
    2) The loss is monotonously increasing with the number of pruned parameters
    3) The loss is almost constant under global magnitude pruning if epsilon is not chosen too high.
    
    TAMADE finds the highest epsilon, such that the loss does not increase by more 
    than loss * `tolerated_relative_loss_increase`.
    
    TAMADE returns the pruned model together with epsilon and the number of steps taken.
    
    Example Usage:
        bsp = Tamade(validation_set, batch_size, device, world_size, rank, ...)
        loss_before = bsp.utils.evaluate_loss_on_set(ddp_model, bsp.evaluation_dataloader, device, world_size, rank)
        pruned_model, epsilon, steps = bsp(distributed_trainer.ddp_model)
        loss_after = bsp.utils.evaluate_loss_on_set(pruned_model, bsp.evaluation_dataloader, device, world_size, rank)
    """
    def __init__(self, 
                 device, world_size, rank, batch_size,
                 validation_set,
                 evaluation_sample_size = 100,
                 tolerated_relative_loss_increase = 1e-3, 
                 epsilon_min = 0.0, epsilon_max = 0.5, tol=1e-7, max_steps=50, 
                 ):
        """Initialize with validation data and search parameters for pruning threshold.
        
        Args:
            device: Device to run computations on.
            world_size: Number of distributed processes.
            rank: Rank of current process.
            batch_size: Batch size for evaluation.
            validation_set: Dataset to evaluate pruning quality.
            evaluation_sample_size: Number of samples to use for evaluation. Will be 
                adjusted to be divisible by world_size. Defaults to 100.
            tolerated_relative_loss_increase: Maximum allowed relative increase in loss. Defaults to 1e-3.
            epsilon_min: Minimum pruning threshold to consider. Defaults to 0.0.
            epsilon_max: Maximum pruning threshold to consider. Defaults to 0.5.
            tol: Tolerance for binary search convergence. Defaults to 1e-7.
            max_steps: Maximum number of binary search steps. Defaults to 50.
        """
        evaluation_sample_size = max(evaluation_sample_size // world_size * world_size, world_size)
        self.validation_set = validation_set
        self.batch_size = batch_size
        self.evaluation_sample_size = evaluation_sample_size
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.tolerated_relative_loss_increase = tolerated_relative_loss_increase
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.tol = tol
        self.max_steps = max_steps
        self.utils = PruningUtils()
        self.evaluation_dataloader = "Not yet loaded"

    def __call__(self, ddp_model: DDP, distributed_trainer: DistributedTransformerTrainer):  
        """Execute pruning procedure on provided model.
        
        Args:
            ddp_model: The DistributedDataParallel model to prune.
            distributed_trainer: Trainer managing the distributed training process.
            
        Returns:
            tuple: (pruned_model, epsilon, steps) where:
                - pruned_model: The pruned DDP model
                - epsilon: The optimal pruning threshold found
                - steps: Number of binary search steps taken
        """
        if self.evaluation_dataloader == "Not yet loaded":
            self.load_evaluation_dataloader()
            
        dist.barrier()
        baseline_loss = self.utils.evaluate_loss_on_set(
            ddp_model, self.evaluation_dataloader, self.device, self.world_size, self.rank
        )
        dist.barrier()
        
        pruned_model = MultiLayerTransformerDecoder(distributed_trainer.embedding_dimension, distributed_trainer.num_heads, distributed_trainer.ff_dim, distributed_trainer.num_layers, pmmp=distributed_trainer.args["pmmp"], dev=distributed_trainer.device).to(distributed_trainer.device)
        pruned_model = DDP(pruned_model, device_ids=[distributed_trainer.local_rank], find_unused_parameters=False)
        
        def _f_to_search_on(x):
            """Evaluate loss with global magnitude pruning at threshold x.
            
            Args:
                x: Pruning threshold value.
                
            Returns:
                float: Loss value after pruning with threshold x.
            """
            pruned_model.module.load_state_dict(ddp_model.module.state_dict())
            self.utils.global_magnitude_pruning(pruned_model, x)      
            # dist.barrier()
            y = self.utils.evaluate_loss_on_set(pruned_model, self.evaluation_dataloader, self.device, self.world_size, self.rank)
            # dist.barrier()
            return y
            
        epsilon, steps = self.utils.binary_search_highest_x(
            _f_to_search_on,
            baseline_loss,
            self.tolerated_relative_loss_increase,
            self.epsilon_min,
            self.epsilon_max,
            tol=self.tol,
            max_steps=self.max_steps,
            world_size=self.world_size,
            rank=self.rank
        )
        
        pruned_model.module.load_state_dict(ddp_model.module.state_dict())
        self.utils.global_magnitude_pruning(pruned_model, epsilon)
        print("Pruned model with epsilon: ", epsilon, " after searching for optimal epsilon in ", steps, " steps.")
        
        return pruned_model, epsilon, steps
        
    def load_evaluation_dataloader(self):
        """Prepare dataloader with subset of validation data for evaluation.
        
        Creates a DataLoader with a copy of a subset of the validation data for 
        consistent evaluation during pruning threshold search.
        
        Raises:
            AssertionError: If validation set doesn't have required attributes or size.
        """
        assert hasattr(self.validation_set, "data"), "Validation set must have data"
        assert len(self.validation_set) >= self.evaluation_sample_size, "Validation set must have at least average_over samples"
        
        evaluation_set = copy.deepcopy(self.validation_set)
        evaluation_set.data = evaluation_set.data[:self.evaluation_sample_size]
        self.evaluation_dataloader = DataLoader(evaluation_set, batch_size=self.batch_size, shuffle=False)


class PruningUtils:
    """Utility methods for model pruning and evaluation.
    
    Static methods:
        global_magnitude_pruning(ddp_model, threshold): 
            Prune all parameters of a DDP model smaller than a threshold
        evaluate_loss_on_set(ddp_model, dataloader, device, world_size, rank):
            Evaluate the loss of a model on a dataset
        binary_search_highest_x(f, base_value, perturbation, x_min, x_max, ...):
            Find the highest x such that f(x) increases at most by base_value * perturbation
    """

    def __init__(self):
        """Initialize pruning utilities."""
        pass
        
    @staticmethod
    def global_magnitude_pruning(ddp_model: DDP, threshold: float):
        """Apply global magnitude pruning to model by zeroing parameters below threshold.
        
        Args:
            ddp_model: DistributedDataParallel model to prune.
            threshold: Magnitude threshold below which parameters are zeroed.
        """
        for name, param in ddp_model.module.named_parameters():
            if param.requires_grad:
                with torch.no_grad():
                    mask = param.abs() >= threshold
                    new_param = param * mask.float()
                    param.copy_(new_param)

    @staticmethod
    def evaluate_loss_on_set(ddp_model: DDP, dataloader, device: torch.device, world_size: int, rank: int):
        """Calculate average loss across dataloader with distributed evaluation.
        
        Args:
            ddp_model: DistributedDataParallel model to evaluate.
            dataloader: DataLoader providing batches for evaluation.
            device: Device to run the evaluation on.
            world_size: Number of distributed processes.
            rank: Rank of current process.
            
        Returns:
            float: Average loss across the evaluation dataset.
        """
        ddp_model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Split the batch across GPUs
                local_batch_size = batch.size(0) // world_size
                start_index = rank * local_batch_size
                end_index = start_index + local_batch_size
                local_batch = batch[start_index:end_index].to(device)
                
                input_seq = local_batch[:, :-1]
                target_seq = local_batch[:, 1:]
                
                # self.optimizer.zero_grad() ??
                output = ddp_model(input_seq)
                output = output.contiguous().view(-1, output.size(-1))
                target_seq = target_seq.contiguous().view(-1)
                
                criterion = nn.CrossEntropyLoss()
                loss = criterion(output, target_seq)
                
                if torch.isnan(loss):
                    print(f"NaN loss detected. Rank: {rank}, Batch: {num_batches}")
                    print(f"Output shape: {output.shape}, Target shape: {target_seq.shape}")
                    print(f"Output min/max: {output.min().item()}, {output.max().item()}")
                    print(f"Target min/max: {target_seq.min().item()}, {target_seq.max().item()}")
                
                total_loss += loss.item()
                num_batches += 1
                
        # Aggregate losses from all processes
        dist.all_reduce(torch.tensor([total_loss, num_batches], device=device))
        
        return total_loss / num_batches if num_batches > 0 else 0
        
    @staticmethod
    def binary_search_highest_x(
        f,
        base_value: float,
        perturbation: float,
        x_min: float,
        x_max: float,
        tol=1e-6,
        max_steps: int=50,
        world_size: int=1,
        rank: int=0,
        verbose: bool=False
    ):
        """Find the highest x such that f(x) increases at most by base_value * perturbation.
        
        Uses binary search under the assumption that f is a monotone increasing function.
        
        Args:
            f: Function to evaluate, assumed to be monotonically increasing.
            base_value: Reference value (typically baseline loss).
            perturbation: Maximum allowed relative increase from base_value.
            x_min: Lower bound of search range.
            x_max: Upper bound of search range.
            tol: Tolerance for convergence. Defaults to 1e-6.
            max_steps: Maximum number of binary search steps. Defaults to 50.
            world_size: Number of distributed processes. Defaults to 1.
            rank: Rank of current process. Defaults to 0.
            verbose: Whether to print progress information. Defaults to False.
            
        Returns:
            tuple: (best_x, steps) where:
                - best_x: Highest value of x meeting the constraint
                - steps: Number of binary search steps taken
                
        Raises:
            AssertionError: If x_min >= x_max.
        """
        assert x_min < x_max
        threshold = base_value * (1 + perturbation)
        steps = 0
        best_x = x_min
        
        if verbose and rank == 0:
            print("base_value: ", base_value, " perturbation: ", perturbation, " threshold: ", threshold)
            
        while x_max - x_min > tol:
            steps += 1
            x_mid = (x_min + x_max) / 2          
            y = f(x_mid)
            y = abs(y)
            
            if (x_min + x_max) / 2 == x_min:
                # This condition should not be reached but might be due to finite numerical precision
                return best_x, steps
                
            if y <= threshold:
                best_x = x_mid
                x_min = x_mid
            else:
                x_max = x_mid
                
            if verbose and rank == 0:
                print("For epsilon = ", x_mid, " the loss is ", y, " and the threshold is ", threshold)
                
            if x_max - x_min <= tol:
                break
            elif steps >= max_steps:
                break        
                
        return best_x, steps