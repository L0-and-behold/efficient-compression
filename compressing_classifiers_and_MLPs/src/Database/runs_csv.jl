"""
Functions to handle the runs.csv file in the experiment folder.
This module provides utilities for creating, initializing, and appending to run data
in CSV format for experiment tracking.
"""

using DataFrames

META_FIELD_NAMES = [
    :run_id,
    :experiment_name,
    :timestamp,
    :run_time
]
METRICS_FIELD_NAMES = [
    :final_accuracy,
    :final_accuracy_trainset,
    :final_accuracy_valset,
    :final_accuracy_testset,
    :final_loss,
    :final_loss_trainset,
    :final_loss_valset,
    :final_loss_testset,
    :final_l0_loss,
    :final_sigma,
    :small_weights,
    :l0_norm,
    :byte_size_compression_ratio,
    :description_length,
    :teacher_student_size_divergence,
]

"""
    initialize_runs_csv(path_to_db::String, experiment_name::String, args)

Initialize a new runs.csv file for tracking experiment runs.

# Arguments
- `path_to_db::String`: Base path to the database directory
- `experiment_name::String`: Name of the experiment
- `args`: Object containing experiment parameters as fields

Creates a DataFrame with columns for all parameters in `args`, metadata fields,
and metric fields, then writes it to a CSV file if it doesn't already exist.
"""
function initialize_runs_csv(path_to_db::String, experiment_name::String, args)
    params_field_values = [x for x in fieldnames(typeof(args))]
    field_names = vcat(params_field_values, META_FIELD_NAMES, METRICS_FIELD_NAMES)
    df = DataFrame([Symbol(field) => Union{Missing, Any}[] for field in field_names])
    #save df to csv
    if !isfile(joinpath(expanduser(path_to_db), experiment_name, "runs.csv"))
        CSV.write(joinpath(expanduser(path_to_db), experiment_name, "runs.csv"), df)
    end
end

"""
    initialize_single_run_df(args) -> DataFrame

Create a DataFrame for a single experiment run.

# Arguments
- `args`: Object containing experiment parameters as fields

# Returns
- A DataFrame with one row, containing columns for all parameters in `args`,
  metadata fields, and metric fields, initialized with missing values.
"""
function initialize_single_run_df(args)
    params_field_values = [x for x in fieldnames(typeof(args))]
    field_names = vcat(params_field_values, META_FIELD_NAMES, METRICS_FIELD_NAMES)
    df = DataFrame([Symbol(field) => Union{Missing, Any}[] for field in field_names])
    df = DataFrame([Symbol(field) => Union{Missing, Any}[missing] for field in names(df)])
    return df
end

"""
    append_run_to_csv!(path_to_db::String, experiment_name::String, single_run_df)

Append a single run's data to the experiment's runs.csv file.

# Arguments
- `path_to_db::String`: Base path to the database directory
- `experiment_name::String`: Name of the experiment
- `single_run_df`: DataFrame containing data for a single run

Reads the existing runs.csv file, appends the new run data, and writes
the updated DataFrame back to the file. Handles missing values by converting
them to empty strings for CSV writing.
"""
function append_run_to_csv!(path_to_db::String, experiment_name::String, single_run_df)
    df_csv = CSV.read(joinpath(expanduser(path_to_db), experiment_name, "runs.csv"), DataFrame)
    df_csv = vcat(df_csv, single_run_df)
    single_run_df = DataFrames.transform(single_run_df, DataFrames.names(single_run_df) .=> ByRow(x -> something(x, "")), renamecols=false) # replace missing with "" for CSV.write
    CSV.write(joinpath(expanduser(path_to_db), experiment_name, "runs.csv"), df_csv)
end

"""
    log_params(df, args) -> DataFrame

Add parameters from args to the last row of a DataFrame.

# Arguments
- `df`: DataFrame to update
- `args`: Object containing parameter values as fields

# Returns
- The updated DataFrame with parameter values filled in the last row

Iterates through all fields in `args` and sets the corresponding values
in the last row of the DataFrame.
"""
function log_params(df, args)
    for field in fieldnames(typeof(args))
        df[end, Symbol(field)] = getfield(args, field)
    end
    return df
end