import os

class Database:
    """
    A Database is a folder at the location 'path_to_db' (variable)
    The Database contains experiments, which are folders with the name 'unique_experiment_name' (variable)
    Each experiment contains 
    - a 'runs.csv' file, which is a CSV file that contains the results of each run in the experiment
    - a 'description.md' file, which is intended to contain a description of the experiment
    - an 'artifacts' folder, which contains the artifacts of each run in the experiment
        - each run has a folder with a unique name, which contains the artifacts of that run

    path_to_db
    |--- unique_experiment_name
    ... |--- runs.csv
    ... |--- description.md
    ... |--- artifacts
    ... ... |--- unique_run_id
    ... ... ... |--- α_state.JSON
    ... ... ... |--- model-before_FT.jld2
    ... ... ... |--- other_artifact
    ... ... ... ...
    """

    def __init__(self, path_to_database):
        self.path = path_to_database
        self.path = os.path.expanduser(self.path)
        self.assert_database_exists()

    def assert_database_exists(self):
        path = os.path.abspath(self.path)
        if not os.path.exists(path):
            error_message = "Database not found at " + str(path)
            print("Debug - Path:", path)
            print("Debug - Error message:", error_message)
            raise FileNotFoundError(error_message)