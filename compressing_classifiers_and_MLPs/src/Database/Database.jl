"""
Module to be used in place of MLFlow for the purpose of this project. 

This module is used to create and manage experiments and runs, as well as to store and retrieve artifacts.

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
... ... ... |--- ...
... ... ... ...
"""

module Database

    using DataFrames, CSV, UUIDs

    using JLD2, JSON

    include("../TrainingTools/TrainingTools.jl")
    using .TrainingTools

    include("db_structure.jl")
    export create_experiment, 
        create_run,
        search_run,
        get_artifact_path,
        artifact_exists,
        get_artifact_folder

    include("runs_csv.jl")
    export initialize_runs_csv, 
        initialize_single_run_df, 
        append_run_to_csv!, 
        log_params

    include("prerun.jl")
    export has_been_run_before, 
        pretrained_models_exist, 
        fetch_pretrained_model_state, 
        fetch_pretrained_optimizer_state

end