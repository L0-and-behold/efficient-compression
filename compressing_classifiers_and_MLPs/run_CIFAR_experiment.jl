"""
    run_an_experiment.jl

Executes a parametric experiment as a batch of runs with configurable variables.
Supports sub-batch execution via --num_sub_batches and --sub_batch command line arguments.
"""

#####
# Header
#####

using Pkg; Pkg.activate("."); # using Revise

using CUDA, ArgParse, Suppressor, Optimisers, ParameterSchedulers
using Lux: cpu_device, gpu_device

using CompressingClassifiersMLPs

using CompressingClassifiersMLPs.Config
using CompressingClassifiersMLPs.TrainingArguments
using CompressingClassifiersMLPs.OptimizationProcedures
using CompressingClassifiersMLPs.DatasetsModels
using CompressingClassifiersMLPs.BatchRun

# using CompressingClassifiersMLPs.TrainingArguments: TrainArgs
# using CompressingClassifiersMLPs.OptimizationProcedures: PMMP_procedure, RL1_procedure, DRR_procedure, layerwise_procedure, Lenet_MLP, Lenet_5, Lenet_5_Caffe, VGG
# using CompressingClassifiersMLPs.DatasetsModels: MNIST_data, CIFAR_data
# using CompressingClassifiersMLPs.BatchRun: do_batch_run, get_sub_batch, single_run_routine_classifier, single_run_routine_teacherstudent, parse_resume_checkpoint

#####
# Experiment setup
#####

"""
TrainArgs object `args` holds all parameters that define an experimental run.
This structure encapsulates the configuration needed to reproduce
a specific training run.
"""

args = TrainArgs{Float32}()

"""
Output location configuration.
Results stored at: <project_root>/experiments/<experiment_name>/
"""

# Directory for saving results
path_to_db = joinpath(pwd(), "experiments")
experiment_name = "CIFAR_5_val_acc"

"""
Run routine selector: 
- single_run_routine_teacherstudent: For MLP compression
- single_run_routine_classifier: For MNIST/CIFAR classification
"""

single_run_routine = single_run_routine_classifier

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
    :β,
    :initial_p_value,
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

batch = [
    (DRR_procedure, 3.33f-06, 5f0, 0f0, Int(5)), # DRR pruned
    (RL1_procedure, 0f0, 0f0, 0f0, Int(5)), # DRR and RL1 vanilla counterpart
    (RL1_procedure, 1.56f-05, 0f0, 0f0, Int(5)), # RL1 pruned
    (PMMP_procedure, 6.67f-05, 0f0, 1f0, Int(0)), # PMMP pruned
    (RL1_procedure, 0f0, 0f0, 0f0, Int(0)), # PMMP vanilla
]

"""
Fixed parameter configuration.
Overrides defaults from TrainArgs.jl for all runs in this experiment.
See README.md for documentation of the funcitionality of each argument.
"""

# Fixed arguments for all runs
args.architecture = VGG
args.dataset = CIFAR_data
args.train_batch_size = 500
args.smoothing_window = 20
args.min_epochs = 30
args.max_epochs = 300
args.finetuning_min_epochs = 10
args.finetuning_max_epochs = 50
args.train_set_size = "see dataset"
args.val_set_size = "see dataset"
args.val_batch_size = "val_set_size"
args.test_set_size = "see dataset"
args.test_batch_size = "test_set_size"
args.noise = 0f0
args.prune_window = 100
shrinking = true
args.shrinking_from_deviation_of = 1e-2
args.gauss_loss = false
args.dev = gpu_device()
args.tamade_val_acc_tolerance = 0.01f0 #prune to at most 1.0pp absolute val acc drop

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
args.resume_checkpoint_id = parse_resume_checkpoint()

do_batch_run(path_to_db, experiment_name, single_run_routine, args, variables, batch; break_if_one_run_errors=break_if_one_run_errors)

println("Experiment completed.")