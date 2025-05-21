"""
    run_an_experiment.jl

Executes a parametric experiment as a batch of runs with configurable variables.
Supports sub-batch execution via --num_sub_batches and --sub_batch command line arguments.
"""

#####
# Header
#####

using Pkg
Pkg.activate(".")

using Revise, ArgParse
using Lux: cpu_device
using Suppressor

# We suppress warnings on docstring replacements. Turn off for development.
@suppress begin
    include("src/OptimizationProcedures/OptimizationProcedures.jl")
    using .OptimizationProcedures: PMMP_procedure,
        RL1_procedure,
        DRR_procedure,
        layerwise_procedure, 
        Lenet_MLP, 
        Lenet_5, 
        Lenet_5_Caffe, 
        VGG
end

include("src/TrainArgs.jl")
include("src/BatchRun/BatchRun.jl")
include("src/DatasetsModels/DatasetsModels.jl")
using .DatasetsModels: MNIST_data, CIFAR_data
using .BatchRun: do_batch_run, 
    get_sub_batch,
    single_run_routine_classifier,
    single_run_routine_teacherstudent  

#####
# Experiment setup
#####

"""
TrainArgs object `args` holds all parameters that define an experimental run.
This structure encapsulates the configuration needed to reproduce
a specific training run.
"""

args = TrainArgs(; T=Float32)

"""
Output location configuration.
Results stored at: <project_root>/experiment-results/<experiment_name>/
"""

# Directory for saving results
path_to_db = joinpath(pwd(), "experiment-results")
println(homedir())
experiment_name = "example-experiment"

"""
Run routine selector: 
- single_run_routine_teacherstudent: For MLP compression
- single_run_routine_classifier: For MNIST/CIFAR classification
"""

single_run_routine = single_run_routine_teacherstudent

"""
Experimental variables configuration.

The `variables` array defines which parameters will be varied across experimental runs
and establishes the correspondence between positions in the batch tuples and parameter names.
For example:
- If analyzing performance across different regularization strengths: include `:α` 
- If comparing optimization procedures: include `:optimization_procedure`
- If studying initialization impact: include `:seed`

Each Symbol in this array corresponds positionally to values in the batch tuples.
The experiment results will be indexed and can be analyzed/plotted along these dimensions.

Each Symbol corresponds to one field of the args object initiallized from src/TrainArgs.jl
"""

# Arguments that vary throughout the experiment
variables = Symbol[
    :optimization_procedure, 
    :α, 
    :seed,
    ]

"""
The `batch` array contains tuples that define the complete experimental space.
Each tuple represents one experimental configuration, with elements corresponding
positionally to the parameters defined in the `variables` array.

In this configuration:
- First position (variables[1] = :optimization_procedure): Optimization procedure to use
- Second position (variables[2] = :α): Regularization strength
- Third position (variables[3] = :seed): Random initialization seed

The total number of runs equals the product of the number of values for each variable.
Here: 3 procedures × 2 alpha values × 2 seeds = 12 total experimental runs.

This grid-based approach enables systematic exploration of the parameter space.
"""

# Values for the varying arguments
batch = Tuple[
    (procedure, alpha, seed)
        for procedure in [DRR_procedure, RL1_procedure, PMMP_procedure]
        for alpha in Float32[1e-4, 1e-5]
        for seed in Int[0, 1]
]

"""
Fixed parameter configuration.
Overrides defaults from TrainArgs.jl for all runs in this experiment.
See README.md for documentation of the funcitionality of each argument.
"""

# Fixed arguments for all runs
args.architecture_teacher = [2, 5, 8, 1]
args.architecture_student = [2, 25, 25, 1]
args.max_epochs = 5000
args.min_epochs = 500
args.prune_window = 5
args.finetuning_max_epochs = 1000
args.train_set_size = 100
args.train_batch_size = args.train_set_size
args.val_set_size = 100
args.val_batch_size = args.val_set_size
args.test_set_size = 100
args.test_batch_size = args.test_set_size
args.smoothing_window = 50
args.dev = cpu_device()
args.dataset = "teacher_student"
args.architecture = "teacher_student"
args.lr = 5f-4
args.gauss_loss = false

"""
Error handling configuration.
When true, terminates all runs if any single run fails.
"""

break_if_one_run_errors = true

#####
# Execute the experiment
#####

"""
Execution handler for batch processing.
Supports parallelization through command-line sub-batch specification.
"""

# If provided via command line arguments, run only a subset of the batch
experiment_name, batch = get_sub_batch(experiment_name, batch)

do_batch_run(path_to_db, experiment_name, single_run_routine, args, variables, batch; break_if_one_run_errors=break_if_one_run_errors)

println("Experiment completed.")