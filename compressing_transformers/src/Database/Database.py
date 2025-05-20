import os

class Database:
    """A class representing a database of experiments.
    
    A Database is a folder structure that organizes experiments and their associated
    data in a hierarchical manner.
    
    Structure:
        path_to_db/
        ├── unique_experiment_name/
        │   ├── runs.csv             # Contains results of each run in the experiment
        │   ├── description.md       # Contains description of the experiment
        │   └── artifacts/           # Contains artifacts from each experimental run
        │       └── unique_run_id/   # Folder for each individual run
        │           ├── α_state.JSON
        │           ├── model-before_FT.jld2
        │           └── other_artifact
    
    Attributes:
        path (str): Absolute path to the database directory
    """

    def __init__(self, path_to_database):
        """Initialize the Database object.
        
        Args:
            path_to_database (str): Path to the database directory
        """
        self.path = path_to_database
        self.path = os.path.expanduser(self.path)
        self.assert_database_exists()

    def assert_database_exists(self):
        """Verify that the database directory exists.
        
        Raises:
            FileNotFoundError: If the database directory doesn't exist
        """
        path = os.path.abspath(self.path)
        if not os.path.exists(path):
            error_message = "Database not found at " + str(path)
            print("Debug - Path:", path)
            print("Debug - Error message:", error_message)
            raise FileNotFoundError(error_message)