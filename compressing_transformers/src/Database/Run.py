import os
import random
import string
import pandas as pd
import time
import torch
import itertools

from src.Transformer.TransformerDecoder import MultiLayerTransformerDecoder
from src.Transformer.TransformerConfig import TransformerConfig


class Run:
    """Represent a run in an experiment.
    
    A run is a unique instance of a model training process, stored as a folder
    in the 'artifacts' directory of an experiment containing run artifacts.
    
    Attributes:
        experiment: The experiment object that the run belongs to.
        id: Unique identifier for the run (matches run folder name).
        info: DataFrame with run information (matches row in `runs.csv`).
            None if run has not been loaded.
        path: Absolute path to the run folder.
        pmmp: Boolean flag for model configuration.
    
    Usage:
        - Create a run: Initialize with experiment, then call `create_run()`
        - Load existing run: Initialize with experiment, then call `load_run(run_id)`
        - Delete a run: Call `delete_run()`
        - Check artifact existence: `run.artifact_exists("loss.png")`
        - Get artifact path: `run.get_artifact_path("loss.png")`
        - Load model: `run.load_model(device)`
        - Load optimizer: `run.load_optimizer(model)`
    """
    def __init__(self, experiment, pmmp=False):
        self.experiment = experiment
        self.start_time = time.time()
        self.id = None
        self.path = None
        self.info = None
        self.prefix = "run"
        self.pmmp = pmmp

    def __str__(self):
        return f"Run {self.id} in experiment {self.experiment.name}"
    
    def __print__(self):
        print(self.__str__())

    def load_run(self, run_id):
        """Load an existing run by ID.
        
        Args:
            run_id: Identifier of the run to load.
            
        Returns:
            self: The Run object with loaded information.
            
        Raises:
            FileNotFoundError: If run with the given ID doesn't exist.
        """
        path = os.path.join(self.experiment.path, "artifacts", run_id)
        if os.path.exists(path):
            self.id = run_id
            self.path = path
            self.info = self.fetch_run_info()
            return self
        else:
            raise FileNotFoundError(f"Run not found at {path}")

    def create_run(self):
        """Create a new run with a unique ID.
        
        Returns:
            self: The Run object with the newly created run information.
            
        Raises:
            FileExistsError: If unable to create a run folder.
        """
        self.id = self.generate_unique_run_id()
        self.path = self.create_run_folder()
        self.assert_run_exists()
        return self

    def delete_run(self):
        """Delete the current run folder if it exists."""
        if os.path.exists(self.path):
            os.rmdir(self.path)

    def generate_unique_run_id(self):
        """Generate a unique run ID not currently in use.
        
        Returns:
            str: A unique run identifier.
            
        Raises:
            ValueError: If a unique ID can't be generated after 100 attempts.
        """
        local_random = random.Random(int(time.time() * 1000))
        existing_ids = set(os.listdir(os.path.join(self.experiment.path, "artifacts")))
        for i in range(100):
            unique_postfix = ''.join(local_random.choices(string.ascii_lowercase + string.digits, k=4))
            new_id = f"{self.prefix}-{unique_postfix}"
            if new_id not in existing_ids:
                return new_id
        raise ValueError("Could not generate unique run id after 100 attempts.")

    def artifact_exists(self, filename):
        """Check if an artifact exists in the run folder.
        
        Args:
            filename: Name of the artifact file to check.
            
        Returns:
            bool: True if the artifact exists, False otherwise.
        """
        return os.path.exists(os.path.join(self.path, filename))

    def get_artifact_path(self, filename):
        """Get the absolute path to an artifact in the run folder.
        
        Args:
            filename: Name of the artifact file.
            
        Returns:
            str: Absolute path to the artifact.
            
        Raises:
            FileNotFoundError: If the artifact doesn't exist.
        """
        exists = self.artifact_exists(filename)
        if not exists:
            raise FileNotFoundError(f"Artifact {filename} not found in run {self.id}.")
        return os.path.join(self.path, filename)

    def create_run_folder(self):
        """Create a folder for the run.
        
        Returns:
            str: Path to the created folder.
            
        Raises:
            FileExistsError: If folder creation fails after 20 attempts.
        """
        for i in range(20):
            path = os.path.join(self.experiment.path, "artifacts", self.id)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                return path
            self.id = self.generate_unique_run_id() # If folder already exists, generate new id and try again
        raise FileExistsError(f"Run folder could not be created after 20 attempts. Path: {path}")
    
    def fetch_run_info(self):
        """Fetch run information from the experiment's runs.csv file.
        
        Returns:
            DataFrame: Information about this run.
            
        Raises:
            FileNotFoundError: If run ID not found in runs.csv.
            ValueError: If multiple entries with same run ID found.
        """
        df = pd.read_csv(os.path.join(self.experiment.path, "runs.csv"))
        info = df[df["run_id"] == self.id]
        if len(info) == 0:
            raise FileNotFoundError(f"Run {self.id} not found in experiment {self.experiment.name}.")
        elif len(info) > 1:
            raise ValueError(f"Multiple runs with id {self.id} found in experiment {self.experiment.name}.")
        return info

    def assert_run_exists(self):
        """Verify that the run folder exists.
        
        Raises:
            FileNotFoundError: If the run folder doesn't exist.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Run not found at {self.path}, experiment {self.experiment.name}, database {self.experiment.database.path}.")
        
    def load_model(self, device):
        """Load a model from the run folder.
        
        Args:
            device: The device to load the model onto (CPU/GPU).
            
        Returns:
            MultiLayerTransformerDecoder: The loaded model.
            
        Raises:
            ValueError: If run info hasn't been loaded.
        """
        if self.info is None:
            raise ValueError("Trying to load model without run info loaded.")

        transformer_config = self.info["transformer_config"].values[0]
        config = TransformerConfig(transformer_config)()
        embedding_dim = config["embedding_dim"]
        num_heads = config["num_heads"]
        num_layers = config["num_layers"]
        ff_dim = config["ff_dim"]

        model = MultiLayerTransformerDecoder(embedding_dim, num_heads, ff_dim, num_layers, pmmp=self.pmmp).to(device)
        model_path = os.path.join(self.path, "model.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    def load_optimizer(self, model):
        """Load an optimizer from the run folder.
        
        Args:
            model: The model whose parameters should be optimized.
            
        Returns:
            torch.optim.Adam: The loaded optimizer.
            
        Raises:
            ValueError: If run info hasn't been loaded.
        """
        if self.info is None:
            raise ValueError("Trying to load optimizer without run info loaded.")

        try:
            lr = self.info["learning_rate"].values[0]
        except KeyError:
            print("Learning rate not found in run info. Using default learning rate of 1e-4.")
            lr = 1e-4

        if self.pmmp:
            optimizer = torch.optim.Adam(itertools.chain(model.parameters(), model.parameters_w(), model.parameters_p(), model.parameters_u()), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer_path = os.path.join(self.path, "optimizer.pth")
        optimizer.load_state_dict(torch.load(optimizer_path))
        return optimizer