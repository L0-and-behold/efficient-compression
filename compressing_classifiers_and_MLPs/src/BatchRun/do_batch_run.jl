"""
    do_batch_run(
        path_to_db::String,
        experiment_name::String,
        single_run_routine::Function,
        args,
        variable_names::Array{Symbol},
        batch_of_values::Union{Vector{<:Tuple}, DeviceIterator};
        break_if_one_run_errors::Bool = true
    )

Run a batch of experiments with different parameter values.

# Arguments
- `path_to_db::String`: Path to the database directory where results will be stored
- `experiment_name::String`: Name of the experiment
- `single_run_routine::Function`: Function that performs a single experiment run
- `args`: Struct containing all parameters for the experiment
- `variable_names::Array{Symbol}`: Names of variables to be modified in each run
- `batch_of_values::Union{Vector{<:Tuple}, DeviceIterator}`: Collection of value tuples to set for each run
- `break_if_one_run_errors::Bool=true`: Whether to stop the batch if one run fails

# Example
```julia
variable_names = [:learning_rate, :batch_size]
batch_values = [(0.001, 64), (0.01, 128), (0.001, 256)]
do_batch_run("./results", "experiment1", run_training, params, variable_names, batch_values)
```
"""

using JLD2
using ..Checkpointer

function do_batch_run(
    path_to_db::String,
    experiment_name::String,
    single_run_routine::Function,
    args,
    variable_names::Array{Symbol},
    batch_of_values::Union{Vector{<:Tuple}, DeviceIterator};
    break_if_one_run_errors::Bool = true
)
    @assert length(variable_names) == length(batch_of_values[1])
    
    # Init checkpoint — ID printed immediately so it appears in logs even if run crashes early
    checkpoint = CheckpointManager(
        args.use_checkpoints,
        CheckpointMetadata(path=joinpath(path_to_db, experiment_name, "checkpoints")),
        CheckpointContent(args=args)
    )
    println("Checkpoint ID: $(checkpoint.metadata.checkpoint_id)")

    # Explicit resume: only when --resume_checkpoint id was passed
    if !isnothing(args.resume_checkpoint_id)
        if !args.use_checkpoints
            @warn "resume_checkpoint_id provided but use_checkpoints=false — enabling checkpointing."
            args.use_checkpoints = true
            checkpoint = CheckpointManager(true, checkpoint.metadata, checkpoint.content)
        end
        println("Resuming from checkpoint: $(args.resume_checkpoint_id)")
        metadata, content = load_checkpoint_by_id(checkpoint.metadata.path, args.resume_checkpoint_id, args)
        checkpoint.metadata = metadata
        checkpoint.content = content
        single_run_routine(path_to_db, experiment_name, args, variable_names, checkpoint)
        println("✓ Resumed run completed successfully. Continuing with batchrun.")
    else
        println("No checkpoint — starting fresh run")
    end

    # Normal batch execution

    # randomize order of runs such that we can run the script in parallel
    # one run is not run twice, this is ensured by 'has_been_run_before' logic in the 'single_run_routine'
    # warning: too many workers might lead to issues when reading/writing to disc at the same time
    shuffle!(batch_of_values)

    # Database setup
    mkpath(path_to_db)
    create_experiment(path_to_db, experiment_name, args)
    initialize_runs_csv(path_to_db, experiment_name, args)
    println("Data will be stored in $path_to_db / $experiment_name. Starting with batch of $(length(batch_of_values)) runs with variables: $(variable_names)")

    summary = "Summary : batch of $(length(batch_of_values)) runs with variables: $(variable_names) \n"

    # Loop over variable combinations
    for (run_idx, value_tupel) in enumerate(batch_of_values)
        

        # Set the values of the variables for this run
        local_args = deepcopy(args)
        for (i, symbol) in enumerate(variable_names)
            field_type = typeof(getfield(local_args, symbol))
            try
                # Try to convert the value to the expected type
                converted_value = value_tupel[i] isa Number ? 
                    convert(field_type, value_tupel[i]) : 
                    value_tupel[i]
                setfield!(local_args, symbol, converted_value)
            catch e
                error("Failed to convert $(value_tupel[i]) to type $field_type for field $symbol: $e")
            end
        end

        # Print info to console
        summary *= "Run $(run_idx) with values: $(value_tupel)"
        message = "Starting run with values: "
        for (i, symbol) in enumerate(variable_names)
            message *= "$symbol: $(value_tupel[i]) "
        end
        println(message)

        # Do training and save results for this run
        if break_if_one_run_errors
            single_run_routine(path_to_db, experiment_name, local_args, variable_names, checkpoint)
            summary *= " - success \n"
        else
            try
                single_run_routine(path_to_db, experiment_name, local_args, variable_names, checkpoint)
                summary *= " - success \n"
            catch e
                println("Run failed with error: $e")
                println("Continue with next run")
                summary *= " - failed \n"
            end
        end
    end
    println("Batch finished!")
    println(summary)
    flush(stdout); flush(stderr)
end