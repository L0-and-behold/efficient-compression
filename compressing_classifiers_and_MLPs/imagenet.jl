using Pkg; Pkg.activate("."); using Revise

flush(stdout); flush(stderr)

using CUDA, TOML, ArgParse, Suppressor, Optimisers, ParameterSchedulers
using Lux: gpu_device

flush(stdout); flush(stderr)

using CompressingClassifiersMLPs

flush(stdout); flush(stderr)

using CompressingClassifiersMLPs.Config
using CompressingClassifiersMLPs.TrainingArguments
using CompressingClassifiersMLPs.OptimizationProcedures
using CompressingClassifiersMLPs.DatasetsModels
using CompressingClassifiersMLPs.BatchRun

flush(stdout); flush(stderr)

#####
# Experiment setup
#####
args = TrainArgs{Float32}()

# Load configuration
path_to_db, imagenet_path, imagenet_preprocessed_path = load_imagenet_config()

# set experiment name
experiment_name = "resnet_all_methods_dev"

# Function defining a single run of training, metric calculation, and result saving
#    either .._classifier or .._teacherstudent
single_run_routine = single_run_routine_classifier

# Arguments that vary throughout the experiment
variables = Symbol[
:optimization_procedure, 
:α, 
:β, 
:NORM,
:initial_p_value,
:u_value_multiply_factor,
:seed,
:shrinking,
]

# Values for the varying arguments
batch = Tuple[]

vanilla_baseline = [
    (
        RL1_procedure, 0f0, 0f0, false, 0f0, 0f0, 1, false
    )
]
append!(batch, vanilla_baseline)

RL1_runs = [
    (
        RL1_procedure, 1f-5, 0f0, false, 0f0, 0f0, 1, true
    )
]

DRR_runs = [
    (
        RL1_procedure, 1f-5, 5f0, false, 0f0, 0f0, 1, true
    )
]

PMMP_runs = [
    (
        PMMP_procedure, 1f-5, 0f0, false, 1f0, 1f0, 1, true
    )
]

# Fixed arguments for all runs

args.architecture = resnet50
args.dataset = imagenet_data_function(imagenet_preprocessed_path, construct_chunked_dataloaders)
args.train_set_size = 1_281_024
args.val_set_size = 50000
args.test_set_size = 50000
args.train_batch_size = 128  # must be divisible by chunk_size in imagenet_preprocessed_path
args.val_batch_size = 128    # same constraint

args.min_epochs = 85 # 85
args.max_epochs = 85 # 85
args.finetuning_min_epochs = 10
args.finetuning_max_epochs = 10
args.lr = 0.05f0
args.schedule = Step(args.lr, 0.1f0, 30)
args.optimizer = lr -> Optimisers.OptimiserChain(
    Optimisers.WeightDecay(0.0001f0),
    Optimisers.Momentum(lr, 0.9f0)
)

args.smoothing_window = 5
args.prune_window = args.min_epochs # prune only once
args.shrinking_from_deviation_of = 1e-2
args.multiply_mask_after_each_batch = false # should be true??

args.noise = 0f0
args.gauss_loss = false
args.dev = gpu_device()
args.converge_val_loss = false

args.debug = false

flush(stdout); flush(stderr)

#####
# Execute the experiment
#####

# If provided via command line arguments, run only a subset of the batch
experiment_name, batch = get_sub_batch(experiment_name, batch)

do_batch_run(path_to_db, experiment_name, single_run_routine, args, variables, batch; break_if_one_run_errors=true)