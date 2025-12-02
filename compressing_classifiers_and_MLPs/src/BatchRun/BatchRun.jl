module BatchRun
    using Suppressor, DataFrames, Dates, CSV, Plots, CUDA, Random, CSV, ArgParse, JLD2
    using Dates: now
    import Lux, LuxCore, PlotlyJS, Optimisers
    import Lux: DeviceIterator, AutoZygote
    import Lux.Training: compute_gradients

    using ..OptimizationProcedures: is_saturated, 
        scale_alpha_rho!, 
        generate_tstate, 
        accuracy, 
        logitcrossentropy, 
        scale_alpha_rho!, 
        generate_tstate, 
        setup_data_teacher_and_student, 
        plot_data_teacher_and_student, 
        plot_weights
    using ..Database: create_experiment, 
        initialize_runs_csv, 
        create_run, 
        initialize_single_run_df, 
        log_params, 
        get_artifact_folder, 
        append_run_to_csv!, 
        has_been_run_before,
        pretrained_models_exist
    using ..TrainingTools: save_train_state, 
        load_train_state
    using ..TrainingArguments
    # using ..DatasetsModels
    using ..Checkpointer

    include("procedures_runs_helpers.jl")
    export do_small_run_to_trigger_precompilation, 
        save_CSV, 
        do_and_save_plot, 
        do_plot, 
        do_and_save_plots

    include("single_run_routine_classifier.jl")
    export single_run_routine_classifier

    include("single_run_routine_teacherstudent.jl")
    export single_run_routine_teacherstudent

    include("command_line_arguments.jl")
    export get_sub_batch

    include("do_batch_run.jl")
    export do_batch_run
end # module BatchRun