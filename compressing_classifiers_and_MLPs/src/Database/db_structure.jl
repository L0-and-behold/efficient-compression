"""
    Module containing functions to handle the (folder) structure in the database.
    
    This module provides utilities for creating experiments, managing runs, and handling artifacts
    in a structured database system.
"""

"""
    create_experiment(path_to_db::String, experiment_name::String, args)

Create a new experiment directory structure in the database.

# Arguments
- `path_to_db::String`: Path to the database root folder
- `experiment_name::String`: Name of the experiment to create
- `args`: Arguments that will be used to initialize the runs.csv file

Creates a folder for the experiment with a description.md file, an artifacts folder,
and initializes a runs.csv file to track experiment runs.
"""
function create_experiment(path_to_db::String, experiment_name::String, args)
    experiment_folder = joinpath(expanduser(path_to_db), experiment_name)
    if !isdir(experiment_folder)
        mkpath(experiment_folder)
        description = """
            # $experiment_name 
            Add description for experiment $experiment_name here
            
            ## Motivation
            ## Setup
            ## Results
            """
        open(joinpath(experiment_folder, "description.md"), "w") do f
            write(f, description)
        end
        mkpath(joinpath(experiment_folder, "artifacts"))
        initialize_runs_csv(path_to_db, experiment_name, args)
    end
end

"""
    create_run(path_to_db::String, experiment_name::String) -> String

Create a new run for an experiment and return a unique run ID.

# Arguments
- `path_to_db::String`: Path to the database root folder
- `experiment_name::String`: Name of the experiment

# Returns
- A unique run ID string for the newly created run

Creates a folder structure for the run with a unique ID and returns that ID.
"""
function create_run(path_to_db::String, experiment_name::String)
    function _random_string()
        unique_id = string(UUIDs.uuid4())
        unique_id_parts = split(unique_id, "-")
        unique_id = unique_id_parts[1]
        return "run-"*unique_id
    end
    random_run_id = _random_string()
    while isdir(joinpath(expanduser(path_to_db), experiment_name, random_run_id))
        random_run_id = _random_string()
    end
    mkpath(joinpath(expanduser(path_to_db), experiment_name, "artifacts", random_run_id))

    return String(random_run_id)
end

"""
    search_run(path_to_db::String, experiment_name::String, query::Dict{Symbol, Any}) -> Vector{String}

Search for runs matching specific criteria within an experiment.

# Arguments
- `path_to_db::String`: Path to the database root folder
- `experiment_name::String`: Name of the experiment to search in
- `query::Dict{Symbol, Any}`: Dictionary of key-value pairs to match against run parameters

# Returns
- Vector of run IDs that match the query criteria

# Example
```julia
query = Dict{Symbol, Any}()
query[:lr] = 0.01
query[:epochs] = 5000
run_ids = search_run(db_path, "my_experiment", query)
```
"""
function search_run(path_to_db::String, experiment_name::String, query::Dict{Symbol, Any})
    df = CSV.read(joinpath(expanduser(path_to_db), experiment_name, "runs.csv"), DataFrame)
    # filter all rows that match the query
    csv_is_good = true
    for (key, value) in query
        df, no_missing_rows = filter_missing_rows(df, key)
        csv_is_good = csv_is_good && no_missing_rows

        # convert Functions to strings, as they are stored as strings in the csv file
        if value isa Function
            value = string(value)
        end

        if value isa String
            df = filter(row -> string(row[key]) == value, df)
        # allow for small numerical differences due to Float32, Float64, Int etc. conversion during saving and loading
        elseif value isa Number 
            df = filter(row -> isapprox(row[key], value), df)
        else
            df = filter(row -> row[key] == value, df)  
        end
    end
    
    if !csv_is_good
        @warn "The .csv file has missing values where there should be none. Check the file for bad rows."
    end

    # return the run_id column as a vector
    run_ids = String.(df[!, :run_id])
    return Vector(run_ids)
end

"""
    filter_missing_rows(df::DataFrame, key::Symbol) -> Tuple{DataFrame, Bool}

Filter out rows with missing values in a specified column.

# Arguments
- `df::DataFrame`: The DataFrame to filter
- `key::Symbol`: The column name to check for missing values

# Returns
- Tuple containing:
  - Filtered DataFrame with no missing values in the specified column
  - Boolean indicating whether all rows were kept (true) or some were removed (false)

Missing values can occur when the CSV file has bad entries, for example when different 
threads access the file at the same time.
"""
function filter_missing_rows(df::DataFrame, key::Symbol)
    if any(ismissing, df[!, key])
        df = filter(row -> !ismissing(row[key]), df)
        return df, false
    end
    return df, true
end

"""
    get_artifact_folder(path_to_db::String, experiment_name::String, run_id::String) -> String

Get the path to the artifact folder for a specific run, creating it if it doesn't exist.

# Arguments
- `path_to_db::String`: Path to the database root folder
- `experiment_name::String`: Name of the experiment
- `run_id::String`: ID of the run

# Returns
- Path to the artifact folder for the specified run
"""
function get_artifact_folder(path_to_db::String, experiment_name::String, run_id::String)
    if !isdir(joinpath(expanduser(path_to_db), experiment_name, "artifacts", run_id))
        mkdir(joinpath(expanduser(path_to_db), experiment_name, "artifacts", run_id))
    end
    return joinpath(expanduser(path_to_db), experiment_name, "artifacts", run_id)
end

"""
    artifact_exists(path_to_db::String, experiment_name::String, run_id::String, artifact_name::String) -> Bool

Check if a specific artifact exists for a run.

# Arguments
- `path_to_db::String`: Path to the database root folder
- `experiment_name::String`: Name of the experiment
- `run_id::String`: ID of the run
- `artifact_name::String`: Name of the artifact to check

# Returns
- `true` if the artifact exists, `false` otherwise
"""
function artifact_exists(path_to_db::String, experiment_name::String, run_id::String, artifact_name::String)::Bool
    artifact_path = get_artifact_folder(path_to_db, experiment_name, run_id)
    exists = isfile(joinpath(artifact_path, artifact_name))
    return exists
end

"""
    get_artifact_path(path_to_db::String, experiment_name::String, run_id::String, artifact_name::String) -> String

Get the full path to a specific artifact for a run.

# Arguments
- `path_to_db::String`: Path to the database root folder
- `experiment_name::String`: Name of the experiment
- `run_id::String`: ID of the run
- `artifact_name::String`: Name of the artifact

# Returns
- Full path to the specified artifact
"""
function get_artifact_path(path_to_db::String, experiment_name::String, run_id::String, artifact_name::String)
    return joinpath(get_artifact_folder(path_to_db, experiment_name, run_id), artifact_name)
end