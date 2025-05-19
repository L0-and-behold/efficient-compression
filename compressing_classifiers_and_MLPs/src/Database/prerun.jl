"""
    has_been_run_before(path_to_db::String, experiment_name::String, args, variables::Vector; check_for_dublicate_runs=true)::Bool

Check if a run with the same hyperparameters has been run before.
Use in a loop over hyperparameters to check if a run has been done before and if so, skip that run.

# Arguments
- `path_to_db::String`: Path to the database
- `experiment_name::String`: Name of the experiment
- `args`: Object containing the hyperparameters
- `variables::Vector`: Vector of variables to check
- `check_for_dublicate_runs::Bool=true`: Whether to check for duplicate runs

# Returns
- `::Bool`: Whether a run with the same hyperparameters has been run before
"""
function has_been_run_before(path_to_db::String, experiment_name::String, args, variables::Vector; check_for_dublicate_runs=true)::Bool
    query = Dict{Symbol, Any}()
    for variable in variables
        if !(variable isa Symbol)
            variable = Symbol(variable)
        end

        value = getfield(args, variable)
        if value isa Function
            query[variable] = string(value)
        else
            query[variable] = value
        end
    end
    run_ids = search_run(path_to_db, experiment_name, query)

    if length(run_ids) > 1 && check_for_dublicate_runs
        @warn("These runs have the same hyperparameters (unwanted dublicates?): $run_ids")
        return true
    else
        return length(run_ids) > 0
    end
end    

"""
    pretrained_models_exist(path_to_db, experiment_name, args, variables)::Bool

Check if run with same hyperparameters, except for ϵ, has been done before.
If yes, use the pretrained model from that prior run to continue training with a new ϵ (fine tuning).

# Arguments
- `path_to_db`: Path to the database
- `experiment_name`: Name of the experiment
- `args`: Object containing the hyperparameters
- `variables`: Vector of variables to check

# Returns
- `::Bool`: Whether a pretrained model exists
"""
function pretrained_models_exist(path_to_db, experiment_name, args, variables)

    query = _pretrained_query(args, variables)

    run_ids = search_run(path_to_db, experiment_name, query)

    if length(run_ids) == 0
        return false
    end

    filename = get_artifact_path(path_to_db, experiment_name, run_ids[1], "model-before_FT.jld2")
    if !isfile(filename)
        @warn("model-before_FT.jld2 not found for run $(run_ids[1]) even though pretrained model should exist.")
        return false
    end

    return true
end

"""
    fetch_pretrained_optimizer_state(path_to_db, experiment_name, args, variables)

Fetch the optimizer state from a prior run with the same hyperparameters, except for ϵ.
In the case that a model has been trained with the same hyperparameters, except for ϵ, 
we only need to do fine tuning. The optimizer state is fetched from the prior run.

# Arguments
- `path_to_db`: Path to the database
- `experiment_name`: Name of the experiment
- `args`: Object containing the hyperparameters
- `variables`: Vector of variables to check

# Returns
The optimizer state from a prior run
"""
function fetch_pretrained_optimizer_state(path_to_db, experiment_name, args, variables)

    query = _pretrained_query(args, variables)

    run_ids = search_run(path_to_db, experiment_name, query)
    filename = get_artifact_path(path_to_db, experiment_name, run_ids[1], "optimizer.jld2")
    if !isfile(filename)
        @warn("File not found: $filename. Will create a new optimizer according to the arguments: $args.optimizer")
        return TrainingTools.get_optimizer(args)
    end
    model_state = JLD2.load(filename, "opt")
    return model_state
end


"""
    fetch_pretrained_model_state(path_to_db, experiment_name, args, variables)

Fetch the model state from a prior run with the same hyperparameters, except for ϵ.
In the case that a model has been trained with the same hyperparameters, except for ϵ,
we only need to do fine tuning. The model state is fetched from the prior run.

# Arguments
- `path_to_db`: Path to the database
- `experiment_name`: Name of the experiment
- `args`: Object containing the hyperparameters
- `variables`: Vector of variables to check

# Returns
The model state from a prior run
"""
function fetch_pretrained_model_state(path_to_db, experiment_name, args, variables)

    if !pretrained_models_exist(path_to_db, experiment_name, args, variables)
        @error("Pretrained model does not exist for this run. Make sure to only call this function if `pretrained_models_exist` returns true.")
    end

    query = _pretrained_query(args, variables)

    run_ids = search_run(path_to_db, experiment_name, query)
    filename = get_artifact_path(path_to_db, experiment_name, run_ids[1], "model-before_FT.jld2")

    model_state = JLD2.load(filename, "model_state")
    
    return model_state
end

"""
    _pretrained_query(args, variables)::Dict{Symbol, Any}

Helper function to create a query for the database to find a pretrained model.

# Arguments
- `args`: Object containing the hyperparameters
- `variables`: Vector of variables to include in the query

# Returns
- `::Dict{Symbol, Any}`: Query dictionary for the database
"""
function _pretrained_query(args, variables)

    query = Dict{Symbol, Any}()
    for variable in variables
        if !(variable isa Symbol)
            variable = Symbol(variable)
        end
        # when this function is used, ϵ is likely a variable, but should not be used as a filter
        if variable == :ϵ
            continue
        end
        query[variable] = getfield(args, variable)
    end
    # we do not want runs that have used a pretrained model itself, since we are looking for the pretrained model
    query[:used_pretrained_model] = false
    return query
end
