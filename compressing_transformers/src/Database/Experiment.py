import os
import pandas as pd

from src.Database.field_names import META_FIELD_NAMES, METRICS_FIELD_NAMES
from src.Database.Run import Run

class Experiment:
    """Represent an experiment in a database.
    
    An experiment is a collection of runs that are related to each other.
    It consists of a folder in the database, which contains a 'runs.csv' file,
    a 'description.md' file, and an 'artifacts' folder.
    
    Usage:
        - Create an experiment by initializing with a database and a new unique experiment name.
        - Initialize the 'runs.csv' file by calling 'init_csv' with a dictionary of parameters.
        - Load an experiment by initializing with a database and an existing experiment name.
        - Search for runs using the 'search_runs' method with a query dictionary.
        - Prune empty runs using the 'prune_empty_runs' method.
    """
    def __init__(self, database, experiment_name):
        self.database = database
        self.name = experiment_name
        self.path = self.find_or_create_experiment()
        self.assert_experiment_exists()

    def find_or_create_experiment(self):
        path = os.path.join(self.database.path, self.name)

        def _create_experiment(path):
            os.makedirs(path, exist_ok=True)
            os.makedirs(os.path.join(path, "artifacts"), exist_ok=True)
        
        if os.path.exists(path):
           pass
        else:
            _create_experiment(path)
        return path
    
    def assert_experiment_exists(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Experiment not found at {self.path} in database {self.database.path}.")

    def search_runs(self, query):
        """Search for runs matching the specified criteria.
        
        Args:
            query: Dictionary with key-value pairs specifying search criteria.
                  Keys must match columns in the 'runs.csv' file.
        
        Returns:
            list: Run objects that match the query criteria.
        """
        df = pd.read_csv(os.path.join(self.path, "runs.csv"))
        for (key, value) in query.items():
            df = df[df[key] == value]
        run_ids = df["run_id"].tolist()
        runs = []
        for run_id in run_ids:
            runs.append(Run(self).load_run(run_id))
        return runs

    def init_csv(self, args):
        if os.path.exists(os.path.join(self.path, "runs.csv")):
            return
        else:
            parameter = list(args.keys())
            df = pd.DataFrame(columns=[META_FIELD_NAMES + parameter + METRICS_FIELD_NAMES])
            df.to_csv(os.path.join(self.path, "runs.csv"), index=False)
        
    def prune_empty_runs(self):
        """Remove empty run folders that don't correspond to valid run IDs.
        
        Scans the artifacts directory for empty folders whose names (run_ids)
        do not appear in the runs.csv file, and deletes them.
        
        Prints progress information during deletion and a summary upon completion.
        """
        artifacts_path = os.path.join(self.path, "artifacts")
        runs_csv_path = os.path.join(self.path, "runs.csv")
        
        df = pd.read_csv(runs_csv_path)
        valid_run_ids = set(df['run_id'])

        deleted_count = 0
        for folder in os.listdir(artifacts_path):
            folder_path = os.path.join(artifacts_path, folder)
            
            # Check if the folder is empty and its name (run_id) is not in valid_run_ids
            if not os.listdir(folder_path) and folder not in valid_run_ids:
                try:
                    os.rmdir(folder_path)
                    deleted_count += 1
                    print(f"Deleted empty folder: {folder_path}")
                except OSError as e:
                    print(f"Error deleting folder {folder_path}: {e}")

        print(f"Pruning complete. Deleted {deleted_count} empty folders.")