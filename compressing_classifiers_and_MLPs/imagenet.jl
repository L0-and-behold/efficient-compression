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
experiment_name = "aug-ft-sweep-v1"

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
:min_epochs,
:max_epochs,
:finetuning_min_epochs,
:finetuning_max_epochs,
]

# aug-ft-sweep-v1: validate RandomResizedCrop augmentation + FT LR fix + higher alphas
# 1 vanilla + 3×RL1 + 3×DRR + 3×PMMP = 10 runs; 7+5 epochs
# Alphas: full-scale-v1 "aggressive" becomes the new floor; explore 3 orders of magnitude above
batch = [
    # Vanilla baseline: 7+0 (no FT)
    (RL1_procedure,  0f0,  0f0, 0f0, 0f0, 7, 7, 0, 0),

    # RL1: 1e-6 (was v1 aggressive) → 1e-5 → 1e-4
    (RL1_procedure,  1f-6, 0f0, 0f0, 0f0, 7, 7, 5, 5),
    (RL1_procedure,  1f-5, 0f0, 0f0, 0f0, 7, 7, 5, 5),
    (RL1_procedure,  1f-4, 0f0, 0f0, 0f0, 7, 7, 5, 5),

    # DRR: 1e-7 (was v1 aggressive) → 1e-6 → 1e-5
    (DRR_procedure,  1f-7, 5f0, 0f0, 0f0, 7, 7, 5, 5),
    (DRR_procedure,  1f-6, 5f0, 0f0, 0f0, 7, 7, 5, 5),
    (DRR_procedure,  1f-5, 5f0, 0f0, 0f0, 7, 7, 5, 5),

    # PMMP u=5: 1e-8 (was v1 aggressive) → 1e-6 → 1e-4
    (PMMP_procedure, 1f-8, 0f0, 1f0, 5f0, 7, 7, 5, 5),
    (PMMP_procedure, 1f-6, 0f0, 1f0, 5f0, 7, 7, 5, 5),
    (PMMP_procedure, 1f-4, 0f0, 1f0, 5f0, 7, 7, 5, 5),
]

# --- full-scale-v1 batch (commented out) ---
# Vanilla: 90+0; compression: 85+1; alphas conservative→best→aggressive from sweep
# (RL1_procedure,  0f0,   0f0, 0f0, 0f0, 90, 90, 0, 0),
# (RL1_procedure,  1f-9,  0f0, 0f0, 0f0, 85, 85, 1, 1), ...

# --- compression-sweep-v2 batch (commented out) ---
# compression_alphas = Float32[1f-6, 1f-7, 1f-8]
# vanilla_baseline = [(RL1_procedure, 0f0, 0f0, 0f0, 0f0, 9, 9, 0, 0)]
# RL1_runs  = [(RL1_procedure,  alpha, 0f0, 0f0, 0f0, 9, 9, 1, 1) for alpha in compression_alphas]
# DRR_runs  = [(DRR_procedure,  alpha, 5f0, 0f0, 0f0, 9, 9, 1, 1) for alpha in compression_alphas]
# PMMP_runs = [(PMMP_procedure, alpha, 0f0, 1f0, 5f0, 9, 9, 1, 1) for alpha in Float32[1f-17, 1f-9]]
# append!(batch, vanilla_baseline); append!(batch, RL1_runs); append!(batch, DRR_runs); append!(batch, PMMP_runs)

# Fixed arguments for all runs

args.seed = 1
args.architecture = resnet50
args.dataset = imagenet_data_function(imagenet_preprocessed_path, construct_chunked_dataloaders)
args.train_set_size = 1_281_024
args.val_set_size = 50000
args.test_set_size = 50000
args.train_batch_size = 128  # must be divisible by chunk_size in imagenet_preprocessed_path
args.val_batch_size = 128    # same constraint

# min/max_epochs and finetuning epochs are set per-run via variables above
args.lr = 0.05f0
args.ρ = 0.0001f0  # use this parameter to controll weight decay
args.optimizer = lr -> Optimisers.OptimiserChain(
    Optimisers.Momentum(lr, 0.9f0)
)
warmup_epochs = 5
step_schedule = Step(args.lr, 0.1f0, 30)
args.schedule = epoch -> epoch <= warmup_epochs ?
    args.lr * Float32(epoch) / Float32(warmup_epochs) :
    step_schedule(epoch - warmup_epochs)

args.smoothing_window = 1000  # disable convergence detection — run exactly min/max_epochs
args.prune_window = 1000      # disable mid-training pruning (shrinking=false anyway)
args.shrinking_from_deviation_of = 1e-2
args.multiply_mask_after_each_batch = false

args.noise = 0f0
args.gauss_loss = false
args.dev = gpu_device()
args.converge_val_loss = false
args.finetuning_converge_val_loss= false
args.shrinking = false
args.NORM = false

args.tamade_calibration_batches = 200  # use 200 batches for TAMADE
args.save_pre_pruning_model = true     # save tstate before TAMADE so 85-ep runs are recoverable
args.skip_precompilation = true
args.debug = false

flush(stdout); flush(stderr)

#####
# Execute the experiment
#####

# If provided via command line arguments, run only a subset of the batch
experiment_name, batch = get_sub_batch(experiment_name, batch)

do_batch_run(path_to_db, experiment_name, single_run_routine, args, variables, batch; break_if_one_run_errors=true)