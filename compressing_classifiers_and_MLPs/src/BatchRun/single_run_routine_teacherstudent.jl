using Dates, CUDA, CSV, DataFrames, Random
using Dates: now
import PlotlyJS, Plots, Lux, Optimisers
import Plots
import Lux: AutoZygote
import Lux.Training: compute_gradients

using .Database: create_experiment, 
    initialize_runs_csv, 
    create_run, 
    initialize_single_run_df, 
    log_params, 
    get_artifact_folder, 
    append_run_to_csv!, 
    has_been_run_before, 
    pretrained_models_exist
using .TrainingTools: save_train_state, 
    load_train_state
using .OptimizationProcedures: scale_alpha_rho!, 
    generate_tstate, 
    setup_data_teacher_and_student, 
    plot_data_teacher_and_student, 
    plot_weights

"""
    single_run_routine_teacherstudent(path_to_db, experiment_name, args, variables)

Conduct a single training run of a teacher-student neural network setup.

This function manages the complete lifecycle of a teacher-student experiment:
setting up models, training, evaluation, and artifact generation. It prevents
duplicate runs by checking if parameters have been used before.

# Arguments
- `path_to_db::String`: Path to the database directory
- `experiment_name::String`: Name of the experiment
- `args`: Configuration parameters for the experiment
- `variables::Vector{Symbol}`: Variables to track for run identification

# Returns
Nothing. Results are saved to the database and artifact folder.
"""
function single_run_routine_teacherstudent(
    path_to_db::String, 
    experiment_name::String, 
    args, 
    variables::Vector{Symbol})

    assertions_teacherstudent(args)
    plotlyjs()
    
    architecture_teacher = args.architecture_teacher
    architecture_student = args.architecture_student
    
    # check whether these args have been run before
    if has_been_run_before(path_to_db, experiment_name, args, variables)
        println("These exact parameters have been run before:")
        for field in fieldnames(typeof(args))
            if field in variables
                println("$field: $(getfield(args, field))")
            end
        end
        return
    end

    run_id = create_run(path_to_db, experiment_name)
    artifact_folder = get_artifact_folder(path_to_db, experiment_name, run_id)
    base_seed = args.seed

    train_set, validation_set, test_set, tstate, loss_fctn, args, teacher_tstate = setup_data_teacher_and_student(args; architecture_teacher=args.architecture_teacher, architecture_student=args.architecture_student, seed_teacher=base_seed+35, seed_student=base_seed+43, seed_train_set=base_seed+1, seed_val_set=base_seed+2, seed_test_set=base_seed+2, teacher_weight_scaling=2, loss_fctn = Lux.MSELoss(), opt = Optimisers.Adam)
    _, _, _, throwaway_tstate, _, _, _ = setup_data_teacher_and_student(args; architecture_teacher=args.architecture_teacher, architecture_student=args.architecture_student, seed_teacher=base_seed+35, seed_student=base_seed+43, seed_train_set=base_seed+1, seed_val_set=base_seed+2, seed_test_set=base_seed+2, teacher_weight_scaling=2, loss_fctn = Lux.MSELoss(), opt = Optimisers.Adam)

    save_network_plots_before_training(tstate, teacher_tstate, train_set, artifact_folder)
   
    start_time = now()

    println("Start training for $run_id with teacher of dimensions '$(architecture_teacher)', and optimization procedure '$(args.optimization_procedure)'")

    # do small training routine to trigger compilation of the involved functions
    do_small_run_to_trigger_precompilation(args.optimization_procedure, throwaway_tstate, train_set, validation_set, test_set, loss_fctn, args)

    # do actual training
    tstate, logs, loss_fctn = args.optimization_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args)

    # save results
    args.logs = Dict{String, Any}() # There is no need to save all logs in the summary csv file
    current_run_df = initialize_single_run_df(args)
    current_run_df = log_params(current_run_df, args)
    current_run_df = log_final_losses(current_run_df, train_set, validation_set, test_set, tstate, loss_fctn, args)
    current_run_df = log_meta_data_and_metrics(current_run_df, run_id, experiment_name, start_time, logs)
    append_run_to_csv!(path_to_db, experiment_name, current_run_df)

    # save plots, .csv files, and trained weights
    artifact_folder = get_artifact_folder(path_to_db, experiment_name, run_id)
    save_CSV_teacherstudent(artifact_folder, logs)
    do_and_save_plots(artifact_folder, logs, args)
    p = plot_weights(tstate)
    Plots.savefig(p, joinpath(artifact_folder, "trained_student_weights" * ".pdf"))
    save_train_state(tstate, tstate.model, Random.GLOBAL_RNG, joinpath(artifact_folder, "train_state.bson"))

    println("Training and saving of results finished for $run_id.")
end
    
############## Helper functions

"""
    assertions_teacherstudent(args)

Validate the configuration parameters for teacher-student training.

Checks that all required parameters have appropriate types and values,
throwing assertions when conditions are not met.

# Arguments
- `args`: The configuration parameters object to validate
"""
function assertions_teacherstudent(args)
    @assert args.architecture_teacher isa Vector{Int64} "The architecture_teacher field in the TrainArgs object should be a vector of integers but got $(args.architecture_teacher)"
    @assert args.architecture_student isa Vector{Int64} "The architecture_student field in the TrainArgs object should be a vector of integers but got $(args.architecture_student)"
    @assert args.optimization_procedure isa Function  "The optimization_procedure field in the TrainArgs object should be a function as in the OptimizationProcedures module but got $(args.optimization_procedure)"

    @assert args.α isa Number "α should be a number but got $(args.α)"
    @assert args.ρ isa Number "ρ should be a number but got $(args.ρ)"
    @assert args.β isa Number "β should be a number but got $(args.β)"
    @assert args.lr > 0 "Learning rate should be positive but got $(args.lr)"
    @assert args.min_epochs >=0 "Minimum epochs should be non-negative but got $(args.min_epochs)"
    @assert args.max_epochs >=0 "Maximum epochs should be non-negative but got $(args.max_epochs)"
    @assert args.finetuning_max_epochs >=0 "Finetuning maximum epochs should be non-negative but got $(args.finetuning_max_epochs)"
    @assert args.smoothing_window >=0 "Smoothing window should be non-negative but got $(args.smoothing_window)"
    @assert args.binary_search_resolution > 0 "Binary search resolution should be positive but got $(args.binary_search_resolution)"
    @assert args.shrinking_from_deviation_of >= 0 "Shrinking from deviation of should be non-negative but got $(args.shrinking_from_deviation_of)"
    @assert args.noise >= 0 "Noise should be non-negative but got $(args.noise)"
    if args.gauss_loss == true
        @assert args.noise > 0 "Noise should be positive when args.gauss_loss true, but got $(args.noise)"
    end
end

"""
    save_network_plots_before_training(tstate, teacher_tstate, train_set, artifact_folder)

Create and save visualizations of the network before training begins.

Generates and saves plots showing teacher-student data relationships and 
network weights to the specified artifact folder.

# Arguments
- `tstate`: Training state of the student model
- `teacher_tstate`: Training state of the teacher model
- `train_set`: The training dataset
- `artifact_folder::String`: Directory where plots will be saved
"""
function save_network_plots_before_training(tstate, teacher_tstate, train_set, artifact_folder)
    p = plot_data_teacher_and_student(tstate, teacher_tstate, train_set, display_plot=false)
    PlotlyJS.savefig(PlotlyJS.plot(p), joinpath(artifact_folder, "teacher_student_data" * ".pdf"))
    p = plot_weights(teacher_tstate)
    Plots.savefig(p, joinpath(artifact_folder, "teacher_weights" * ".pdf"))
    p = plot_weights(tstate)
    Plots.savefig(p, joinpath(artifact_folder, "untrained_student_weights" * ".pdf"))
end

"""
    log_final_losses(run_df, train_set, validation_set, test_set, tstate, loss_fctn, args)

Calculate and log final losses on all datasets.

Computes the loss on training, validation and test datasets after training
completion and records them in the provided DataFrame.

# Arguments
- `run_df::DataFrame`: The DataFrame to store loss values
- `train_set`: The training dataset
- `validation_set`: The validation dataset
- `test_set`: The test dataset  
- `tstate`: The trained model state
- `loss_fctn`: The loss function
- `args`: Configuration parameters

# Returns
- Updated DataFrame with loss values added
"""
function log_final_losses(run_df::DataFrame, train_set, validation_set, test_set, tstate, loss_fctn, args)
    
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
    log_meta_data_and_metrics(run_df, run_id, experiment_name, start_time, logs)

Add experiment metadata and metrics to the run DataFrame.

Logs run identification, timing information, and specialized metrics like 
L0 norm and sigma values if they exist in the logs.

# Arguments
- `run_df::DataFrame`: The DataFrame to update
- `run_id::String`: Unique identifier for the run
- `experiment_name::String`: Name of the experiment
- `start_time::Dates.DateTime`: When the experiment started
- `logs::Dict`: Dictionary containing training metrics and logs

# Returns
- Updated DataFrame with metadata and metrics added
"""
function log_meta_data_and_metrics(run_df::DataFrame, run_id::String, experiment_name::String, start_time::Dates.DateTime, logs::Dict)
    run_df[end, :run_id] = run_id
    run_df[end, :experiment_name] = experiment_name
    run_df[end, :timestamp] = string(start_time)
    run_df[end, :run_time] = logs["total_time"]

    if haskey(logs, "l0_mask") && length(logs["l0_mask"]) > 0
        run_df[end, :l0_norm] = logs["l0_mask"][end]
    end
    if haskey(logs, "sigmas") && length(logs["sigmas"]) > 0
        run_df[end, :final_sigma] = logs["sigmas"][end]
    end
    return run_df
end

"""
    save_CSV_teacherstudent(artifact_folder, logs)

Save training metrics to CSV files in the artifact folder.

Creates separate CSV files for various metrics including losses,
execution times, L0 mask values, and sigma values.

# Arguments
- `artifact_folder::String`: Directory where CSV files will be saved
- `logs::Dict`: Dictionary containing training metrics to save
"""
function save_CSV_teacherstudent(artifact_folder::String, logs::Dict)
    save_CSV(artifact_folder, logs["train_loss"], "train_loss")
    save_CSV(artifact_folder, logs["val_loss"], "val_loss")
    save_CSV(artifact_folder, logs["test_loss"], "test_loss")

    save_CSV(artifact_folder, logs["epoch_execution_time"], "epoch_execution_time")
    save_CSV(artifact_folder, logs["l0_mask"], "l0_mask")
    save_CSV(artifact_folder, logs["sigmas"], "sigmas")
end