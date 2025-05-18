import os
import time
import pandas as pd

from src.Database.field_names import META_FIELD_NAMES, METRICS_FIELD_NAMES

class RunsCSV:
    """
    During training, use this class to handle the logging of run information to a .csv file.

    ## Usage during training:
        
    For each new run, create a new instance of this class with the run object and the args object.
        The args object should contain the arguments that are used during training. The keys of the args object should match the columns in the .csv file - compare with `src/Databse/field_names.py`

    Example usage:
        ```
        run_df = RunsCSV(run, args)

        # model training ...
        
        run_df.log_meta()
        run_df.log_args(args)
        run_df.log_metric("compression_ratio", 0.5)
        run_df.log_metric("loss", 0.01)
        run_df.save_to_CSV()
        ```

    ## Usage during evaluation:

    During evaluation, we are not appending new rows to the .csv file, but rather want to update inidivual cell values in the .csv file i.e. adding new metrics to blank cells in otherwise filled rows.

    For this, use the following methods:
    - `load_args_from_CSV` to fetch the argument values that were used during training
    - `update_cell_in_CSV` overwrite a cell (associated to a metric and a run) in the csv file 
    - to be implemented: add a new metrics-column to the .csv file    
    """
    def __init__(self, run, args):
        self.run = run
        self.experiment = run.experiment
        self.filename = f"{self.experiment.path}/runs.csv"
        self.args = args
        self.df = self.init_df()

    def init_df(self):
        parameter = list(self.args.keys())
        df = pd.DataFrame(columns=META_FIELD_NAMES + parameter + METRICS_FIELD_NAMES)
        return df

    def log_field(self, fieldname, value):
        self.df.at[0, fieldname] = value

    def log_meta(self):
        end_time = time.time()
        self.log_field("run_id", self.run.id)
        self.log_field("experiment_name", self.experiment.name)
        self.log_field("timestamp", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
        self.log_field("run_time", end_time - self.run.start_time)

    def log_args(self, args):
        for key, value in args.items():
            self.log_field(key, value)

    def log_metric(self, fieldname, value):
        self.log_field(fieldname, value)

    def save_to_CSV(self):
        if not os.path.exists(self.filename):
            self.df.to_csv(self.filename, index=False)
        else:
            csv_df = pd.read_csv(self.filename)
            csv_df = pd.concat([csv_df, self.df], ignore_index=True)
            csv_df.to_csv(self.filename, index=False)

    def load_args_from_CSV(self):
        """
        Load the arguments from the csv file into the run_df object.

        Initialize the run_csv object with some args object whose keys are the same as the columns in the csv file.
        Then overwrite the values of the args object with the values from the csv file.
            ```
            args = {
                "transformer_config": "",
                "epochs": "",
            }
            run_df = RunsCSV(run, args)
            run_df.load_args_from_CSV()
            ```
        """

        df = pd.read_csv(self.filename)
        row = df[df["run_id"] == self.run.id]

        if len(row) == 0:
            raise FileNotFoundError(f"Run {self.id} not found in experiment {self.experiment.name}.")
        elif len(row) > 1:
            raise ValueError(f"Multiple runs with id {self.id} found in experiment {self.experiment.name}.")

        for key in self.args.keys():
            self.args[key] = row[key].values[0]

    def update_cell_in_CSV(self, fieldname, value):
        """
        Append a metric to a previously saved run in the csv file.
        Usecase: for a stored model, we want to append calculated metrics, that were not logged during training, to the .csv file.
        The field in the .csv associated with the run (row) and fieldname (column) is updated with the value.
        
        Example:

            ```
            run = Run(experiment).load_run(run_id)
            run_df = RunsCSV(run, args)

            run_df.append_metric_to_previous_run("compression_ratio", 0.5)
            ```
        """
        df = pd.read_csv(self.filename)
        
        # error handeling
        matches = df["run_id"] == self.run.id
        if not matches.any():
            raise FileNotFoundError(f"Run {self.id} not found in experiment {self.experiment.name}.")
        elif matches.sum() > 1:
            raise ValueError(f"Multiple runs with id {self.id} found in experiment {self.experiment.name}.")

        if fieldname not in df.columns:
            raise KeyError(f"Field {fieldname} not found in the csv file.")

        # writing on the .csv file
        df.loc[matches, fieldname] = value
        df.to_csv(self.filename, index=False)
        

    
            

