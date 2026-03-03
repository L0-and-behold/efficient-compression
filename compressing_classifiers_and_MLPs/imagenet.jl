using Pkg; Pkg.activate(".")
using Revise

flush(stdout); flush(stderr)

using CUDA, TOML, ArgParse, Suppressor, Optimisers, ParameterSchedulers
using Lux: gpu_device
using TOML: parsefile

flush(stdout); flush(stderr)

using CompressingClassifiersMLPs

flush(stdout); flush(stderr)

using CompressingClassifiersMLPs.TrainingArguments: TrainArgs
using CompressingClassifiersMLPs.OptimizationProcedures: PMMP_procedure,
RL1_procedure,
DRR_procedure,
VGG,
Lenet_5_Caffe,
Lenet_MLP,
resnet50,
resnet18, 
toy_resnet,
alexnet
using CompressingClassifiersMLPs.DatasetsModels: MNIST_data, 
CIFAR_data, 
imagenet_data_function, 
construct_online_dataloaders, 
construct_chunked_dataloaders, 
construct_toy_dataloaders
using CompressingClassifiersMLPs.BatchRun: do_batch_run, 
get_sub_batch,
single_run_routine_classifier,
single_run_routine_teacherstudent  

flush(stdout); flush(stderr)

#####
# Experiment setup
#####
args = TrainArgs{Float32}()

# Load configuration
@assert isfile("config.toml") "File `config.toml` does not exist or script run from wrong path."
cfg = parsefile("config.toml")
@assert haskey(cfg, "paths") "config file should have [paths] section"
@assert haskey(cfg["paths"], "path_to_db")
@assert haskey(cfg["paths"], "imagenet_path")
@assert haskey(cfg["paths"], "imagenet_preprocessed_path")
path_to_db = cfg["paths"]["path_to_db"]
imagenet_path = cfg["paths"]["imagenet_path"]
imagenet_preprocessed_path = cfg["paths"]["imagenet_preprocessed_path"]

# set experiment name
experiment_name = "resnet_bs128_weightdecay"

# Function defining a single run of training, metric calculation, and result saving
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
:gradient_repetition_factor
]

# Values for the varying arguments
batch = Tuple[]

append!(batch, [
        (
        RL1_procedure, 
        0f0,
        0f0,
        false,
        0f0,
        0f0,
        1,
        false,
        1
        )
        ]
)

# Fixed arguments for all runs

args.dataset = imagenet_data_function(imagenet_preprocessed_path, construct_chunked_dataloaders)

args.architecture = resnet50 #toy_resnet or resnet50
args.delete_neurons = false
args.layerwise_pruning = false
args.smoothing_window = 5   
args.finetuning_min_epochs = 10 # 10
args.finetuning_max_epochs = 10 # 10
args.train_batch_size = 128  # must be divisible by chunk_size in imagenet_preprocessed_path
args.val_batch_size = 128    # same constraint
args.noise = 0f0
args.prune_window = 10 # this should be turned up to inf maybe or at least tuned.
args.shrinking_from_deviation_of = 1e-2
args.gauss_loss = false
args.dev = gpu_device()
args.converge_val_loss = false # this implies val_loss convergence criterium

args.min_epochs = 85 # 90
args.max_epochs = 85 # 90
args.lr = 0.05f0
args.optimizer = lr -> Optimisers.OptimiserChain(
    Optimisers.WeightDecay(0.0001f0),
    Optimisers.Momentum(lr, 0.9f0)
)
args.train_set_size = 1_281_024
args.val_set_size = 50000
args.test_set_size = 50000

args.schedule = Step(0.05f0, 0.1f0, 30)

args.multiply_mask_after_each_batch = false
args.debug = false

break_if_one_run_errors = true

flush(stdout); flush(stderr)

#####
# Execute the experiment
#####

# If provided via command line arguments, run only a subset of the batch
experiment_name, batch = get_sub_batch(experiment_name, batch)

do_batch_run(path_to_db, experiment_name, single_run_routine, args, variables, batch; break_if_one_run_errors=break_if_one_run_errors)


# todo: turn off early stopping for imagenet