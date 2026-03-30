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
path_to_db, imagenet_path, _ = load_imagenet_config()

# set experiment name
experiment_name = "checkpoint-cr-test"#"full-scale-v2"

# TODO
# [ ] debug mode (191172)
# [ ] 3+1 epochs
# [ ] commit checkpoint cr
# [ ] implement LR, rho, cosine schedule, number epochs
# [ ] implement color jitter
# [ ] test debug mode
# [ ] test 3+1
# [ ] maybe little sweep over lr and rho for vanilla otherwise
# [ ] start full-scale-v3 for vanilla and wait

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

# full-scale-v2: FT LR fix + RandomResizedCrop + post-TAMADE acc report + 5 FT epochs
# 1 vanilla + 3×RL1 + 3×DRR + 3×PMMP = 10 runs; 90+0 / 85+5
batch = [
    # Vanilla baseline: 90+0 (no FT)
    # (RL1_procedure,  0f0,   0f0, 0f0, 0f0, 90, 90, 0, 0),

    # (RL1_procedure,  1f-7,  0f0, 0f0, 0f0, 85, 85, 5, 5),
    # (RL1_procedure,  1f-6,  0f0, 0f0, 0f0, 85, 85, 5, 5),
    # (RL1_procedure,  3f-6,  0f0, 0f0, 0f0, 85, 85, 5, 5),

    (DRR_procedure,  1f-8,  5f0, 0f0, 0f0, 3, 3, 1, 1),   # TEST: 3+1 epochs
    # (DRR_procedure,  1f-7,  5f0, 0f0, 0f0, 85, 85, 5, 5),
    # (DRR_procedure,  1f-6,  5f0, 0f0, 0f0, 85, 85, 5, 5),

    # (PMMP_procedure, 1f-6,  0f0, 1f0, 1f0, 85, 85, 5, 5),
    # (PMMP_procedure, 1f-5,  0f0, 1f0, 5f0, 85, 85, 5, 5),
    # (PMMP_procedure, 1f-5,  0f0, 1f0, 1f0, 85, 85, 5, 5),
]

# --- aug-ft-sweep-v1 batch (commented out) ---
# 7+5 ep; validates RandomResizedCrop + FT LR fix; higher alphas than v1
# (RL1_procedure,  0f0,  0f0, 0f0, 0f0, 7, 7, 0, 0),
# (RL1_procedure,  1f-6, 0f0, 0f0, 0f0, 7, 7, 5, 5), ...

# --- full-scale-v1 batch (commented out) ---
# 90+0 / 85+1; had FT LR bug (restarted warmup during FT)
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
args.dataset = imagenet_data_function()
args.train_set_size = 1_281_024
args.val_set_size = 50000
args.test_set_size = 50000
args.train_batch_size = 128  # must be divisible by chunk_size the proprocessed dataset was built with
args.val_batch_size = 128    # same constraint

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
args.debug = true 
args.use_checkpoints = true            # TEST
args.checkpoint_frequency = 1          # TEST: checkpoint every epoch
args.cr_report_window = 1              # TEST: CR report every epoch

args.label_smoothing = true

args.ρ = 0.8f-4  # use this parameter to controll weight decay
args.lr = 0.1f0
args.optimizer = lr -> Optimisers.OptimiserChain(
    Optimisers.Momentum(lr, 0.9f0)
)
# Cosine LR with 5-epoch linear warmup.
# For FT (epoch > args.max_epochs): holds at eta_min so FT gets a stable small LR.
# ParameterSchedulers.CosAnneal exists but args.max_epochs varies per-run, so we inline.
function imagenet_schedule(epoch, args)
    lr      = args.lr
    eta_min = lr * 1f-2       # floor = 1% of peak (0.001 at lr=0.1)
    warmup  = 5
    T       = args.max_epochs  # set per-run via batch variables (85 or 90)
    if epoch <= warmup
        return lr * Float32(epoch) / Float32(warmup)
    elseif epoch <= T
        t     = Float32(epoch - warmup)
        T_eff = Float32(T - warmup)
        return eta_min + (lr - eta_min) * 0.5f0 * (1f0 + cos(Float32(π) * t / T_eff))
    else
        return eta_min  # FT phase: stable small LR (same magnitude as old step-LR end)
    end
end
args.schedule = imagenet_schedule


flush(stdout); flush(stderr)

#####
# Execute the experiment
#####

# If provided via command line arguments, run only a subset of the batch
experiment_name, batch = get_sub_batch(experiment_name, batch)
args.resume_checkpoint_id = parse_resume_checkpoint()

do_batch_run(path_to_db, experiment_name, single_run_routine, args, variables, batch; break_if_one_run_errors=true)