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

experiment_name = "RL1-val-acc-tamade-test"

single_run_routine = single_run_routine_classifier

# Test run for val-acc TAMADE: prune to at most 2% absolute accuracy drop.
# Compare CR/accuracy against v2 α=1e-6 run (loss-based, 3% loss tolerance).
variables = [:α]

batch = [
    (1f-6,),
]

# Fixed arguments for all runs

args.seed = 1
args.architecture = resnet50
args.dataset = imagenet_data_function()
args.train_set_size = 1_281_024
args.val_set_size = 50000
args.test_set_size = 50000
args.train_batch_size = 128
args.val_batch_size = 128

args.lr = 0.09f0
args.ρ = 8f-6

args.smoothing_window = 1000  # disable convergence detection — run exactly min/max_epochs
args.prune_window = 1000
args.shrinking_from_deviation_of = 1e-2
args.multiply_mask_after_each_batch = false

args.noise = 0f0
args.gauss_loss = false
args.dev = gpu_device()
args.converge_val_loss = false
args.finetuning_converge_val_loss = false
args.shrinking = false
args.NORM = false

# with the intend to stop early
args.min_epochs            = 90
args.max_epochs            = 90
args.finetuning_min_epochs = 10
args.finetuning_max_epochs = 10

args.tolerated_relative_loss_increase = 0.03f0  # 3%

args.optimization_procedure = RL1_procedure
args.β = 0f0
args.initial_p_value = 0f0
args.initial_u_value = 0f0
args.scale_alpha_with_lr = false

args.tamade_calibration_batches = 200
args.save_pre_pruning_model = true
args.skip_precompilation = true
args.debug = false
args.use_checkpoints = true
args.checkpoint_frequency = 5
args.cr_report_window = 10

args.label_smoothing = true
args.tamade_val_acc_tolerance = 0.02f0  # max 2% absolute accuracy drop

args.optimizer = lr -> Optimisers.OptimiserChain(Optimisers.Momentum(lr, 0.9f0))
# Cosine LR with 5-epoch linear warmup. eta_min = 1% of peak = matches step-LR endpoint.
# For FT (epoch > args.max_epochs): holds at eta_min.
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