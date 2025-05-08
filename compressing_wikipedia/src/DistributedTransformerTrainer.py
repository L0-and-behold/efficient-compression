import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import itertools

from src.Transformer.TransformerDecoder import MultiLayerTransformerDecoder
from src.Database.Database import Database
from src.Database.Experiment import Experiment
from src.Database.Run import Run
from src.Database.RunsCSV import RunsCSV

class DistributedTransformerTrainer:
    """
    Manager for distributed transformer training across multiple GPUs.
    
    Handles distributed process setup, model initialization, and result tracking.
    """
    def __init__(self, 
                 args: dict, 
                 path_to_database: str, 
                 experiment_name: str, 
                 transformer_config: dict, 
                 seed = 50999, verbose=False, init_process_group=True):
        """
        Initialize trainer with experiment configuration and distributed setup.
        """
        self.args = args
        self.path_to_database = path_to_database
        self.experiment_name = experiment_name

        self.embedding_dimension = transformer_config["embedding_dim"]
        self.num_heads = transformer_config["num_heads"]
        self.ff_dim = transformer_config["ff_dim"]
        self.num_layers = transformer_config["num_layers"]
        self.learning_rate = transformer_config["learning_rate"]
        self.verbose = verbose

        self.init_process_group = init_process_group
        self.seed = seed

        self.world_size, self.rank, self.local_rank, self.device = world_size_rank_device()

        if verbose:
            print(f"Rank {self.rank}: Initializing DistributedTransformerTrainer")
            print(f"Rank {self.rank}: World size: {self.world_size}, Local rank: {self.local_rank}")
            print(f"Rank {self.rank}: Using device: {self.device}")

        self._set_seed()
        self._database_setup()
        self._world_setup()

    def _set_seed(self):
        """
        Set random seeds for reproducibility across all processes.
        """

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(int(self.seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _database_setup(self):
        """
        Initialize database connections and create run tracking objects.
        """        
        self.database = Database(self.path_to_database)
        self.experiment = Experiment(self.database, self.experiment_name)
        self.run = Run(self.experiment).create_run()
        self.run_df = RunsCSV(self.run, self.args)

    def _world_setup(self):
        """
        Configure distributed process environment and initialize process group.
        """        
        os.environ['MASTER_ADDR'] = os.environ['MASTER_IP']
        os.environ['MASTER_PORT'] = '29502'

        if self.init_process_group:
            dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)

        local_rank = self.rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)

        if self.verbose:
            print(f"Rank {self.rank}: Local rank {local_rank}, Device count: {torch.cuda.device_count()}")

    def model_optimizer(self):
        """
        Create or load model and optimizer, with support for pretrained models.
        
        Returns model, DDP-wrapped model, and optimizer.
        """
        torch.manual_seed(self.seed + self.rank) # Different seed for each rank

        # Initialize model & optimizer
        if "use_pretrained_model" in self.args and self.args["use_pretrained_model"]:
            if self.rank == 0:
                print(f"Loading model from experiment {self.args['use_model_from_experiment']}, run {self.args['use_model_from_run']}.")
            prerun_db = self.database
            prerun_exp = Experiment(prerun_db, self.args["use_model_from_experiment"])
            prerun_run = Run(prerun_exp, pmmp=self.args["pmmp"]).load_run(self.args["use_model_from_run"])

            model = prerun_run.load_model(self.device)
            ddp_model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=False)
            
            optimizer = prerun_run.load_optimizer(model)

            if self.args["epochs"] + prerun_run.info["elapsed_epochs"].values[0] != self.args["elapsed_epochs"]:
                raise ValueError(f"elapsed epochs in run {prerun_run.id} do not match the expected elapsed epochs. Expected: {self.args['elapsed_epochs']}, got: {prerun_run.info['elapsed_epochs'].values[0]}+{self.args['epochs']}")
        else:
            if self.rank == 0:
                print("Initializing model from scratch.")
            model = MultiLayerTransformerDecoder(self.embedding_dimension, self.num_heads, self.ff_dim, self.num_layers, pmmp=self.args["pmmp"], initial_p_value=self.args["initial_p_value"], dev=self.device).to(self.device)
            ddp_model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=False)
            if self.args["pmmp"]:
                optimizer = optim.Adam(itertools.chain(model.parameters(), model.parameters_w(), model.parameters_p(), model.parameters_u()), lr=self.learning_rate)
            else:
                optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

            if self.args["elapsed_epochs"] != self.args["epochs"]:
                raise ValueError(f"Elapsed epochs does not equal epochs. Elapsed: {self.args['elapsed_epochs']}, epochs: {self.args['epochs']}")

        return model, ddp_model, optimizer

    def cleanup(self):
        """
        Clean up distributed process group.
        """        
        dist.destroy_process_group()

    ###########

    def print_model_info(self, model):
        """
        Print model architecture details and parameter count.
        """        
        print(f"Training using {self.world_size} GPUs.")
        print(f"Transformer decoder with {self.num_layers} layers, {self.embedding_dimension} embedding dimensions, {self.num_heads} heads, and {self.ff_dim} feedforward dimensions.")
        print(f"Number of parameters: {model.count_parameters()}")

    def save_plot(self, X_data: list, Y_data: list, title: str):
        """
        Create and save a plot with two subplots: one showing all data and another showing the last quarter.
        This function is robust to handle cases with very few data points.
        """
        plotfilename = os.path.join(self.run.path, f"{title}.png")
        
        fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Full data plot
        ax1.plot(X_data, Y_data)
        ax1.set_title(title)
        ax1.set_xlabel("epoch")
        
        # Last quarter of data (or all data if less than 4 points)
        l = len(X_data)
        if l >= 4:
            indx = l // 4
            ax2.plot(X_data[-indx:], Y_data[-indx:])
        else:
            ax2.plot(X_data, Y_data)
        ax2.set_ylabel(title)
        ax2.set_xlabel("epoch")
        ax2.set_title("Last quarter (or all)" if l >= 4 else "All data")
        
        plt.tight_layout()
        plt.savefig(plotfilename)
        plt.close(fig)  # Close the figure to free up memory

        print(f"Plot saved to {plotfilename}")

    def save_csv(self, X_data: list, Y_data: list, title: str):
        """
        Create and save the loged data to a csv file.
        """
        df = pd.DataFrame({title: Y_data}, index=X_data)
        df.to_csv(os.path.join(self.run.path, f"{title}.csv"))
        print(f"Data saved to {os.path.join(self.run.path, f'{title}.csv')}")


######### Helper functions #########

def world_size_rank_device():
    """
    Return world_size, rank, local_rank, and device from environment variables for distributed training.
    Set default values if not provided, such that the code can be run locally without distributed training.
    """

    # Set default values for environment variables if not provided
    os.environ.setdefault('WORLD_SIZE', '1')
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('MASTER_IP', 'localhost')

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    return world_size, rank, local_rank, device
