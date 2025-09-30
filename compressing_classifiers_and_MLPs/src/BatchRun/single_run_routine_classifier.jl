using Dates, CUDA, Plots, CSV, DataFrames, Random
using Dates: now
import Lux, LuxCore
import Lux: AutoZygote
import Lux.Training: compute_gradients

using .Database: create_experiment, initialize_runs_csv, create_run, initialize_single_run_df, log_params, get_artifact_folder, append_run_to_csv!, has_been_run_before
using .TrainingTools: save_train_state, load_train_state
using .OptimizationProcedures: scale_alpha_rho!, generate_tstate, accuracy

"""
    single_run_routine_classifier(path_to_db::String, experiment_name::String, args, variables)

Execute a single training run for a classifier model with the given parameters.

# Arguments
- `path_to_db::String`: Path to the database directory where results will be stored
- `experiment_name::String`: Name of the experiment
- `args`: Object containing all training parameters and configurations
- `variables`: Collection of variable names that are being optimized or varied in this run

# Returns
Nothing, but saves training results, plots, and model state to the database
"""
function single_run_routine_classifier(path_to_db::String, experiment_name::String, args, variables)

    assertions_classifier(args)

    plotlyjs()

    # check whether these args have been run before
    if has_been_run_before(path_to_db, experiment_name, args, variables)
        println("These exact parameters have been run before: ")
        for field in fieldnames(typeof(args))
            if String(field) in variables || field in variables
                println("$field: $(getfield(args, field))")
            end
        end
        return
    end

    run_id = create_run(path_to_db, experiment_name)

    train_set, validation_set, test_set = args.dataset(args.train_batch_size)
    model = args.architecture()

    model_seed = args.seed + 42; loss_fctn = OptimizationProcedures.logitcrossentropy;
    
    start_time = now()

    println("Start training for $run_id with architecture '$(args.architecture)', dataset '$(args.dataset)' and optimization procedure '$(args.optimization_procedure)'")

    # do some training to trigger compilation of the involved functions
    
    throwaway_tstate = generate_tstate(model, model_seed, args.optimizer(args.lr); dev=args.dev)
    try 
        do_small_run_to_trigger_precompilation(args.optimization_procedure, throwaway_tstate, train_set, validation_set, test_set, loss_fctn, args) 
    catch 
        println("Error during precompilation run. Continue with actual training")
    end
    
    # do actual training
    tstate = generate_tstate(model, model_seed, args.optimizer(args.lr); dev=args.dev)

    tstate, logs, loss_fctn = args.optimization_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args)

    # save results
    args.logs = Dict{String, Any}() # There is no need to save all logs in the summary csv file
    current_run_df = initialize_single_run_df(args)
    current_run_df = log_params(current_run_df, args)
    current_run_df = log_final_accuracies_losses(current_run_df, tstate, train_set, validation_set, test_set, loss_fctn, args) 
    current_run_df = log_meta_data_and_metrics(current_run_df, run_id, experiment_name, start_time, logs)
    append_run_to_csv!(path_to_db, experiment_name, current_run_df)

    # save plots and .csv files
    artifact_folder = get_artifact_folder(path_to_db, experiment_name, run_id)
    save_CSV_classifier(artifact_folder, logs)

    do_and_save_plots(artifact_folder, logs, args)
    save_train_state(tstate, model, Random.GLOBAL_RNG, joinpath(artifact_folder, "train_state.bson"))

    println("Training and saving of results finished for $run_id.")
end

"""
    assertions_classifier(args)

Verify that the training arguments for a classifier are valid and properly configured.

# Arguments
- `args`: Object containing all training parameters to be validated

# Throws
- `AssertionError`: If any of the required arguments are invalid or improperly configured
"""
function assertions_classifier(args)
    @assert args.dataset isa Function "The dataset field in the TrainArgs object should be a function that returns a tuple of train, validation and test set."
    @assert args.architecture isa Function "The architecture field in the TrainArgs object should be a function that returns an appropriate Lux model."
    @assert args.optimization_procedure isa Function  "The optimization_procedure field in the TrainArgs object should be a function as in the OptimizationProcedures module."

    @assert args.α isa Number 
    @assert args.ρ isa Number 
    @assert args.β isa Number
    @assert args.dev isa Function "The dev field in the TrainArgs object should be a function that moves data to the appropriate device, such as Lux.gpu_device() "
    @assert args.lr > 0 "The learning rate should be a positive number."
    @assert args.min_epochs >=0
    @assert args.max_epochs >=0
    @assert args.finetuning_max_epochs >=0
    @assert args.smoothing_window >=0
    @assert args.binary_search_resolution > 0
    @assert args.shrinking_from_deviation_of >= 0
end

"""
    log_final_accuracies_losses(run_df, tstate, train_set, validation_set, test_set, loss_fctn, args)::DataFrame

Calculate and log the final accuracy and loss values for all datasets.

# Arguments
- `run_df::DataFrame`: DataFrame to store the results
- `tstate`: Training state containing the trained model
- `train_set`: Training dataset
- `validation_set`: Validation dataset
- `test_set`: Test dataset
- `loss_fctn`: Loss function used for training
- `args`: Training arguments

# Returns
- `DataFrame`: Updated DataFrame with final accuracies and losses added
"""
function log_final_accuracies_losses(run_df, tstate, train_set, validation_set, test_set, loss_fctn, args)
    run_df[end, :final_accuracy_trainset] = accuracy(tstate, train_set)
    run_df[end, :final_accuracy_valset] = accuracy(tstate, validation_set)
    run_df[end, :final_accuracy_testset] = accuracy(tstate, test_set)
    
    function loss_on_dataset(dataset)::Number
        vjp = AutoZygote()
        total_loss = zero(args.dtype)
        for batch in dataset
            _, loss, _, _ = compute_gradients(vjp, loss_fctn, batch, tstate)
            total_loss += loss
        end
        return total_loss / args.dtype(length(dataset))
    end

    run_df[end, :final_loss_trainset] = loss_on_dataset(train_set)
    run_df[end, :final_loss_valset] = loss_on_dataset(validation_set)
    run_df[end, :final_loss_testset] = loss_on_dataset(test_set)
    
    return run_df
end

"""
    log_meta_data_and_metrics(run_df::DataFrame, run_id::String, experiment_name::String, start_time::Dates.DateTime, logs::Dict)::DataFrame

Record metadata and performance metrics for a classifier training run.

# Arguments
- `run_df::DataFrame`: DataFrame to store the results
- `run_id::String`: Unique identifier for this run
- `experiment_name::String`: Name of the experiment
- `start_time::Dates.DateTime`: Timestamp when training began
- `logs::Dict`: Dictionary containing training logs and metrics

# Returns
- `DataFrame`: Updated DataFrame with metadata and metrics added
"""
function log_meta_data_and_metrics(run_df::DataFrame, run_id::String, experiment_name::String, start_time::Dates.DateTime, logs::Dict)
    run_df[end, :run_id] = run_id
    run_df[end, :experiment_name] = experiment_name
    run_df[end, :timestamp] = string(start_time)
    run_df[end, :run_time] = logs["total_time"]

    if haskey(logs, "l0_mask") && length(logs["l0_mask"]) > 0
        run_df[end, :l0_norm] = logs["l0_mask"][end]
    end
    return run_df
end

"""
    save_CSV_classifier(artifact_folder, logs::Dict{String, Any})

Save training logs for a classifier model to CSV files.

# Arguments
- `artifact_folder`: Directory path where CSV files will be saved
- `logs::Dict{String, Any}`: Dictionary containing various training metrics and logs
"""
function save_CSV_classifier(artifact_folder, logs::Dict{String, Any})
    save_CSV(artifact_folder, logs["train_loss"], "train_loss")
    save_CSV(artifact_folder, logs["test_accuracy"], "test_accuracy")
    save_CSV(artifact_folder, logs["val_loss"], "val_loss")
    save_CSV(artifact_folder, logs["validation_accuracy"], "validation_accuracy")
    save_CSV(artifact_folder, logs["sigmas"], "sigmas")
    # save_CSV(artifact_folder, logs["ascent_loss"], "ascent_loss")
    save_CSV(artifact_folder, logs["l0_mask"], "l0_mask")
    save_CSV(artifact_folder, logs["test_loss"], "test_loss")
    save_CSV(artifact_folder, logs["epoch_execution_time"], "epoch_execution_time")
end 