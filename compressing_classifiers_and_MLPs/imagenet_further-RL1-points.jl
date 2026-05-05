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
# Alpha sweep — RL1 and DRR combined (maybe later PMMP as well)
#
# Vanilla val acc per epoch (job 191578_3, cosine LR, rho=8e-6, label smoothing): 72.2% <- this is our top-1 vanilla value      
#   [sub 1] RL1  α=1.5e-6  FINISHED  at FT-5 val=71.044% (Δ-1.2pp)  CR=74.1% ie. 3.86  279704_1
#   [sub 2] RL1  α=2.5e-6  FINISHED  at FT-3 val=71.102 % (Δ-1.1pp) CR=85.2% ie. 6.78  279704_2 OOM → 279849 OOM → 286444
#   [sub 3] RL1  α=2.1e-6  STOPPED (time constraints)  ep63 val=63.26% CR@ep55=86.2%  287969_3 OOM → 290295 OOM → 291286
#   [sub 4] RL1  α=2.2e-6  STOPPED  (time constraints) ep68 val=64.4% (Δ-3.5pp)  CR@ep50=85.2%  287969_4 OOM → 291287
#   [sub 5] RL1  α=2.3e-6  STOPPED (time constraints and not as promising)  ep33 val=53.7% (Δ-8.7pp)  CR@ep30=87.3%  287969_5
#   [sub 6] RL1  α=2.4e-6  STOPPED (time constraints)  ep15 val=49.9% (Δ-7.0pp)  CR@ep15=87.0%  287969_6 OOM → 290213
#####

args = TrainArgs{Float32}()

# Load configuration
path_to_db, imagenet_path, _ = load_imagenet_config()

experiment_name = "further-RL1-points"

single_run_routine = single_run_routine_classifier

variables = [:optimization_procedure, :α]

batch = [
    (RL1_procedure, 1.5f-6),   # sub 1
    (RL1_procedure, 2.5f-6),   # sub 2
    (RL1_procedure, 2.1f-6),   # sub 3
    (RL1_procedure, 2.2f-6),   # sub 4
    (RL1_procedure, 2.3f-6),   # sub 5
    (RL1_procedure, 2.4f-6),   # sub 6
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
args.multiply_mask_after_each_batch = false

args.noise = 0f0
args.gauss_loss = false
args.dev = gpu_device()
args.converge_val_loss = false
args.finetuning_converge_val_loss = false
args.shrinking = false
args.NORM = false

args.min_epochs            = 90
args.max_epochs            = 90
args.finetuning_min_epochs = 10
args.finetuning_max_epochs = 10

args.tamade_calibration_batches = 200
args.tamade_val_acc_tolerance = 0.013f0 #prune to at most 1.3pp absolute val acc drop

# DRR-specific
args.β = 5f0
# PMMP-specific
args.initial_p_value = 1f0
args.initial_u_value = 1f0

args.save_pre_pruning_model = true
args.skip_precompilation = true
args.debug = false
args.use_checkpoints = true
args.checkpoint_frequency = 5
args.cr_report_window = 5

args.label_smoothing = true

args.optimizer = lr -> Optimisers.OptimiserChain(Optimisers.Momentum(lr, 0.9f0))
# Cosine LR with 5-epoch linear warmup. eta_min = 1% of peak.
# FT phase (epoch > max_epochs): holds at eta_min.
function imagenet_schedule(epoch, args)
    lr      = args.lr
    eta_min = lr * 1f-2
    warmup  = 5
    T       = args.max_epochs
    if epoch <= warmup
        return lr * Float32(epoch) / Float32(warmup)
    elseif epoch <= T
        t     = Float32(epoch - warmup)
        T_eff = Float32(T - warmup)
        return eta_min + (lr - eta_min) * 0.5f0 * (1f0 + cos(Float32(π) * t / T_eff))
    else
        return eta_min
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
