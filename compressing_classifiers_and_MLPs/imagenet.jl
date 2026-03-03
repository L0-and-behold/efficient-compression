using Pkg; Pkg.activate(".")
# Pkg.resolve(); Pkg.instantiate()
using Revise

flush(stdout); flush(stderr)

using CUDA, TOML, ArgParse, Suppressor, Optimisers, ParameterSchedulers

flush(stdout); flush(stderr)
using Lux: gpu_device
using TOML: parsefile

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
println("Using experiment data path: ", path_to_db)
println("Using ImageNet path: ", imagenet_path)

# set experiment name
experiment_name = "resnet"

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
alphas = Float32[0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.07, 0.15, 0.3, 0.7, 1.5, 3.0, 6.0, 12.0, 25.0, 50.0, 100.0, 215.0, 500.0, 1000.0]./45000
seeds = Int[0,1,2,3,4,5]

additional_alphas = Float32[]
append!(additional_alphas, alphas)
# append!(additional_alphas,
#     Float32[0.0001, 0.0013, 0.0016, 3.4, 3.8, 4.2, 4.8, 5.3, 2000.0, 4000.0, 12e3, 36e3, 1e5]./45000
# )
#     Float32[0.0001, 0.1, 0.2, 0.5, 1.0, 2.0, 3.4, 4.8]./45000
# )

vanilla_runs = [
(
    RL1_procedure, 
    0f0, 
    0f0, 
    false,
    0f0,
    1f0,
    seed,
    false,
    1
)
for seed in seeds
]
append!(batch, vanilla_runs)

# DRR_runs = [
#     (
#         DRR_procedure, 
#         α,
#         5f0,
#         false,
#         0f0,
#         1f0,
#         seed,
#         true,
#         1
#     )
#     for α in additional_alphas
#     for seed in seeds
# ]
# append!(batch, DRR_runs)

# RL1_runs = [
#     (
#         RL1_procedure, 
#         α, 
#         0f0, 
#         false,
#         0f0,
#         1f0,
#         seed,
#         true,
#         1
#     )
#     for α in additional_alphas
#     for seed in seeds
# ]
# append!(batch, RL1_runs)

# add PMMP runs here

# FPP_runs= [
#     (
#         FPP_procedure, 
#         α, 
#         0f0, 
#         false,
#         1f0,
#         1f0,
#         seed,
#         shrink,
#         gradient_rep_factor
#     )
#     for α in additional_alphas
#     for seed in seeds
#     for shrink in [true,false]
#     for gradient_rep_factor in [1,5]
# ]
# append!(batch, FPP_runs)

# Fixed arguments for all runs

args.dataset = imagenet_data_function(imagenet_preprocessed_path, construct_chunked_dataloaders)

args.architecture = resnet50 #toy_resnet # resnet50
args.delete_neurons = false
args.layerwise_pruning = false
args.smoothing_window = 5   
args.finetuning_min_epochs = 10
args.finetuning_max_epochs = 10
args.train_set_size = "see dataset"
args.val_set_size = "see dataset"
args.test_set_size = "see dataset"
args.train_batch_size = 160
args.val_batch_size = 160
args.test_batch_size = 160
args.noise = 0f0
args.prune_window = 10
args.shrinking_from_deviation_of = 1e-2
args.gauss_loss = false
args.dev = gpu_device()
args.converge_val_loss = true # this implies val_loss convergence criterium

args.min_epochs = 90 # 100
args.max_epochs = 90 # 100
args.optimizer = lr -> Momentum(lr, 0.9f0)
# args.optimizer = lr -> Optimisers.OptimiserChain( # if weight decay (L2 regularization) is desired
#     Optimisers.WeightDecay(0.0005f0),     # L2 regularization
#     Optimisers.Momentum(lr, 0.9f0)        # SGD + Momentum
# )
args.lr = 0.05f0
args.train_set_size = 1_281_024 # 1_281_167
args.val_set_size = 50000
args.test_set_size = 50000

batches_per_epoch = length(Base.Iterators.partition(1:args.train_set_size, args.train_batch_size))
decay_epochs = max(1, round(Int64, 100_000 / batches_per_epoch))
args.schedule = Step(
args.lr,            # Initial learning rate
0.1f0,              # Decay factor (multiply by 0.1 = divide by 10)
25 # devide lr by 10 every (-) epochs. 100k iterations for Alexnet
)

args.multiply_mask_after_each_batch = false
args.debug = true # false

args.use_checkpoints = false
# args.checkpoint_dir = joinpath(path_to_db, experiment_name, "checkpoints")
# args.checkpoint_frequency = 100
# args.max_runtime_seconds = 3600 * 23.5 * 100

break_if_one_run_errors = true

flush(stdout); flush(stderr)

#####
# Execute the experiment
#####

# If provided via command line arguments, run only a subset of the batch
experiment_name, batch = get_sub_batch(experiment_name, batch)

do_batch_run(path_to_db, experiment_name, single_run_routine, args, variables, batch; break_if_one_run_errors=break_if_one_run_errors)