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
    """
    Class to represent a run in an experiment. A run is a unique instance of a model training process.

    A run is a folder in the 'artifacts' folder of an experiment, which contains the artifacts of the run.

    Properties:
    - experiment: the experiment that the run belongs to. The experiment is an object of the Experiment class.
    - id: a unique identifier for the run, coinciding with the name of the run folder.
    - info: a dataframe with information about the run. Coincides with the row in the `runs.csv` file that matches the run id. Info is None if the run has not been loaded.
    - path: the path to the run folder (absolute path).

    Create a run by initializing an instance of this class with an experiment, then calling the 'create_run' method.
    Load an existing run by initializing an instance of this class with an experiment, then calling the 'load_run' method.
    Delete a run by calling the 'delete_run' method.

    Check if an artifact exists in the run by calling the 'artifact_exists' method.
        E.g. run.artifact_exists("loss.png")
    Get the path to an artifact in the run by calling the 'get_artifact_path' method.
        E.g. run.get_artifact_path("loss.png")

    Load a model from the run by calling the 'load_model' method. The model is loaded from the 'model.pth' file in the run folder.
    Load an optimizer from the run by calling the 'load_optimizer' method. The optimizer is loaded from the 'optimizer.pth' file in the run folder.
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
        path = os.path.join(self.experiment.path, "artifacts", run_id)
        if os.path.exists(path):
            self.id = run_id
            self.path = path
            self.info = self.fetch_run_info()
            return self
        else:
            raise FileNotFoundError(f"Run not found at {path}")

    def create_run(self):
        self.id = self.generate_unique_run_id()
        self.path = self.create_run_folder()
        self.assert_run_exists()
        return self

    def delete_run(self):
        if os.path.exists(self.path):
            os.rmdir(self.path)

    def generate_unique_run_id(self):
        local_random = random.Random(int(time.time() * 1000))
        existing_ids = set(os.listdir(os.path.join(self.experiment.path, "artifacts")))
        for i in range(100):
            unique_postfix = ''.join(local_random.choices(string.ascii_lowercase + string.digits, k=4))
            new_id = f"{self.prefix}-{unique_postfix}"
            if new_id not in existing_ids:
                return new_id
        raise ValueError("Could not generate unique run id after 100 attempts.")

    def artifact_exists(self, filename):
        return os.path.exists(os.path.join(self.path, filename))

    def get_artifact_path(self, filename):
        exists = self.artifact_exists(filename)
        if not exists:
            raise FileNotFoundError(f"Artifact {filename} not found in run {self.id}.")
        return os.path.join(self.path, filename)

    def create_run_folder(self):
        for i in range(20):
            path = os.path.join(self.experiment.path, "artifacts", self.id)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                return path
            self.id = self.generate_unique_run_id() # If folder already exists, generate new id and try again
        raise FileExistsError(f"Run folder could not be created after 20 attempts. Path: {path}")
    
    def fetch_run_info(self):
        df = pd.read_csv(os.path.join(self.experiment.path, "runs.csv"))
        info = df[df["run_id"] == self.id]
        if len(info) == 0:
            raise FileNotFoundError(f"Run {self.id} not found in experiment {self.experiment.name}.")
        elif len(info) > 1:
            raise ValueError(f"Multiple runs with id {self.id} found in experiment {self.experiment.name}.")
        return info

    def assert_run_exists(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Run not found at {self.path}, experiment {self.experiment.name}, database {self.experiment.database.path}.")
        
    def load_model(self, device):
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