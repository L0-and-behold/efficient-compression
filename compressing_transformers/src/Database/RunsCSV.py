import os
import time
import pandas as pd
from src.Database.field_names import META_FIELD_NAMES, METRICS_FIELD_NAMES

class RunsCSV:
    """A class for logging run information to CSV files during training and evaluation.
    
    During training, this class handles logging of run information to a .csv file.
    During evaluation, it allows updating individual cell values in existing CSV records.
    
    Attributes:
        run: The run object associated with this CSV record.
        experiment: The experiment object from the run.
        filename: Path to the CSV file.
        args: Arguments used during training.
        df: DataFrame to store the run information.
    """
    def __init__(self, run, args):
        """Initialize a RunsCSV instance.
        
        Args:
            run: The run object to be logged.
            args: Dictionary of arguments used during training.
        """
        self.run = run
        self.experiment = run.experiment
        self.filename = f"{self.experiment.path}/runs.csv"
        self.args = args
        self.df = self.init_df()
        
    def init_df(self):
        """Initialize DataFrame with appropriate columns.
        
        Returns:
            pandas.DataFrame: Empty DataFrame with columns for meta fields, parameters, and metrics.
        """
        parameter = list(self.args.keys())
        df = pd.DataFrame(columns=META_FIELD_NAMES + parameter + METRICS_FIELD_NAMES)
        return df
        
    def log_field(self, fieldname, value):
        """Log a value to a specific field in the DataFrame.
        
        Args:
            fieldname: Name of the field to log.
            value: Value to log in the field.
        """
        self.df.at[0, fieldname] = value
        
    def log_meta(self):
        """Log metadata about the run including ID, experiment name, timestamp, and duration."""
        end_time = time.time()
        self.log_field("run_id", self.run.id)
        self.log_field("experiment_name", self.experiment.name)
        self.log_field("timestamp", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
        self.log_field("run_time", end_time - self.run.start_time)
        
    def log_args(self, args):
        """Log all arguments to their respective fields.
        
        Args:
            args: Dictionary of arguments to log.
        """
        for key, value in args.items():
            self.log_field(key, value)
            
    def log_metric(self, fieldname, value):
        """Log a metric value to a specific field.
        
        Args:
            fieldname: Name of the metric field to log.
            value: Value of the metric to log.
        """
        self.log_field(fieldname, value)
        
    def save_to_CSV(self):
        """Save the current DataFrame to the CSV file.
        
        If the file doesn't exist, it creates a new one.
        If the file exists, it appends the new data as a new row.
        """
        if not os.path.exists(self.filename):
            self.df.to_csv(self.filename, index=False)
        else:
            csv_df = pd.read_csv(self.filename)
            csv_df = pd.concat([csv_df, self.df], ignore_index=True)
            csv_df.to_csv(self.filename, index=False)
            
    def load_args_from_CSV(self):
        """Load arguments from CSV file for the current run.
        
        Loads values from the CSV row matching the current run ID and
        updates the args attribute with these values.
        
        Example:
            ```
            args = {
                "transformer_config": "",
                "epochs": "",
            }
            run_df = RunsCSV(run, args)
            run_df.load_args_from_CSV()
            ```
            
        Raises:
            FileNotFoundError: If run ID is not found in the CSV.
            ValueError: If multiple entries with the same run ID are found.
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
        """Update a specific cell in the CSV file for the current run.
        
        Used to append metrics to previously saved runs. This is particularly useful
        during evaluation to update metrics that weren't logged during training.
        
        Args:
            fieldname: Name of the field (column) to update.
            value: New value to set in the cell.
            
        Example:
            ```
            run = Run(experiment).load_run(run_id)
            run_df = RunsCSV(run, args)
            run_df.update_cell_in_CSV("compression_ratio", 0.5)
            ```
            
        Raises:
            FileNotFoundError: If run ID is not found in the CSV.
            ValueError: If multiple entries with the same run ID are found.
            KeyError: If the specified field doesn't exist in the CSV.
        """
        df = pd.read_csv(self.filename)
        # error handling
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