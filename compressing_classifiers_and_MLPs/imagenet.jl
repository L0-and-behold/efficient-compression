using Pkg; Pkg.activate("."); using Revise

flush(stdout); flush(stderr)

using CUDA, ArgParse, Suppressor, Optimisers, ParameterSchedulers
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
experiment_name = "PMMP-u-sweep"

# Function defining a single run of training, metric calculation, and result saving
#    either .._classifier or .._teacherstudent
single_run_routine = single_run_routine_classifier

# Arguments that vary throughout the experiment
variables = Symbol[
:optimization_procedure, 
:α, 
:β, 
:initial_p_value,
:initial_u_value,
]

# Values for the varying arguments
batch = Tuple[]

alphas = [1f-8, 3f-8, 1f-7, 3f-7, 1f-6]
vanilla_baseline = [
    (
        RL1_procedure, 0f0, 0f0, 0f0, 0f0
    )
]

RL1_runs = [
    (
        RL1_procedure, alpha, 0f0, 0f0, 0f0
    )
    for alpha in alphas
]

DRR_runs = [
    (
        DRR_procedure, alpha, 5f0, 0f0, 0f0
    )
    for alpha in alphas
]

PMMP_runs = [
    (
        PMMP_procedure, 1f-14, 0f0, 1f0, u_value
    )
    for u_value in Float32[0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
]

# append!(batch, vanilla_baseline)
# append!(batch, RL1_runs)
# append!(batch, DRR_runs)
append!(batch, PMMP_runs)

# Fixed arguments for all runs

args.seed = 1
args.architecture = resnet50
args.dataset = imagenet_data_function(imagenet_preprocessed_path, construct_chunked_dataloaders)
args.train_set_size = 1_281_024
args.val_set_size = 50000
args.test_set_size = 50000
args.train_batch_size = 128  # must be divisible by chunk_size in imagenet_preprocessed_path
args.val_batch_size = 128    # same constraint

args.min_epochs = 9 # 90 # 85
args.max_epochs = 9 # 90 # 85
args.finetuning_min_epochs = 1
args.finetuning_max_epochs = 1
# args.min_epochs = 90 # 85
# args.max_epochs = 90 # 85
# args.finetuning_min_epochs = 0
# args.finetuning_max_epochs = 0
args.lr = 0.05f0
args.optimizer = lr -> Optimisers.OptimiserChain(
    Optimisers.WeightDecay(0.0001f0),
    Optimisers.Momentum(lr, 0.9f0)
)
warmup_epochs = 5
step_schedule = Step(args.lr, 0.1f0, 25)
args.schedule = epoch -> epoch <= warmup_epochs ?
    args.lr * Float32(epoch) / Float32(warmup_epochs) :
    step_schedule(epoch - warmup_epochs)

args.smoothing_window = args.max_epochs + 1
args.prune_window = args.max_epochs # prune only once
args.shrinking_from_deviation_of = 1e-2
args.multiply_mask_after_each_batch = false

args.noise = 0f0
args.gauss_loss = false
args.dev = gpu_device()
args.converge_val_loss = false
args.finetuning_converge_val_loss= false
args.shrinking = false
args.NORM = false

args.skip_precompilation = true
args.debug = false

flush(stdout); flush(stderr)

#####
# Execute the experiment
#####

# If provided via command line arguments, run only a subset of the batch
experiment_name, batch = get_sub_batch(experiment_name, batch)

do_batch_run(path_to_db, experiment_name, single_run_routine, args, variables, batch; break_if_one_run_errors=true)