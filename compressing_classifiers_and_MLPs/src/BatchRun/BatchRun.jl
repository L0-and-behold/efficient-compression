module BatchRun_Lux

    include("../Database/Database.jl")
    include("../TrainingTools/TrainingTools.jl")
    include("../OptimizationProcedures/OptimizationProcedures.jl")
    include("../TrainArgs.jl")
    include("../DatasetsModels/DatasetsModels.jl")

    include("procedures_runs_helpers.jl")
    export do_small_run_to_trigger_precompilation, save_CSV, do_and_save_plot, do_plot, do_and_save_plots

    include("single_run_routine_classifier.jl")
    export single_run_routine_classifier

    include("single_run_routine_teacherstudent.jl")
    export single_run_routine_teacherstudent

    include("command_line_arguments.jl")
    export get_sub_batch

    include("do_batch_run.jl")
    export do_batch_run

end # module