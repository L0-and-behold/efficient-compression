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
# PMMP alpha × u sweep
#
# Motivation: alpha-sweep-v1 PMMP subs (1e-10 to 1e-7, u=1) show flat CR ~38-40% across
# all epochs — essentially vanilla. Hypothesis: alpha too small to push pp away from its
# pp=1 initial attractor; u controls how strongly p tracks pp once pp moves.
# Testing α ∈ {1e-6, 1e-5, 5e-5} × u ∈ {2, 5}.
# Note: alpha-sweep-v1 sub16 (PMMP 1e-5, u=1) collapsed ~ep20. Retest 1e-5 with u=2/5.
#
# Vanilla val acc per epoch (job 191578_3, cosine LR, rho=8e-6, label smoothing):
# ep:  1     2     3     4     5     6     7     8     9    10
#     12.56 27.60 35.00 38.28 43.72 46.90 51.97 52.45 54.34 53.32
# ep: 11    12    13    14    15    16    17    18    19    20
#     58.02 55.20 57.71 58.81 56.90 60.09 58.96 58.42 61.43 60.15
# ep: 21    22    23    24    25    26    27    28    29    30
#     60.04 57.40 59.17 59.81 60.41 62.02 59.18 60.66 60.60 63.94
# ep: 31    32    33    34    35    36    37    38    39    40
#     64.49 60.30 62.34 61.77 63.75 62.43 61.49 60.42 64.44 64.24
# ep: 41    42    43    44    45    46    47    48    49    50
#     65.30 63.71 65.99 65.05 63.97 66.43 65.66 66.18 66.11 63.16
# ep: 51    52    53    54    55    56    57    58    59    60
#     66.45 67.17 64.84 66.72 63.97 66.43 65.66 66.18 66.11 63.16
# ep: 61    62    63    64    65    66    67    68    69    70
#     66.45 67.17 64.84 66.72 65.97 65.57 67.56 67.52 66.84 66.84
# ep: 71    72    73    74    75    76    77    78    79    80
#     67.41 69.40 69.57 69.57 69.71 71.09 70.82 71.11 70.19 69.93
# ep: 81    82    83    84    85    86    87    88    89    90
#     70.35 70.63 71.77 72.16 72.01  —     —     —     —   72.16(~72.0)
#
# Results summary — vanilla ep20=60.1%, ep30=63.9%, ep40=64.2%, ep50=63.2%, ep70=66.8%, ep90=72.0%
#   [sub 1] PMMP  1e-6  u=2:  RUNNING   ep28 val=63.7% (Δ+3.0pp)  CR@ep25=37.3%  ckpt ep25  260555_1 OOM → 277388
#   [sub 2] PMMP  1e-5  u=2:  DEAD — collapsed ep20→21 (loss 2.6→6.9), val 0.1%  ckpt ep25 useless  260555_2
#   [sub 3] PMMP  5e-5  u=2:  RUNNING   ep13 val=48.3% (Δ-9.4pp)  CR@ep10=69.9%  ckpt ep10  260555_3 OOM → 277389
#   [sub 4] PMMP  1e-6  u=5:  RUNNING   ep28 val=64.5% (Δ+3.8pp)  CR@ep25=41.2%  ckpt ep25  260555_4 OOM → 277390
#   [sub 5] PMMP  1e-5  u=5:  RUNNING   ep30 val=60.7% (Δ-3.2pp)  CR@ep30=74.2%              260555_5
#   [sub 6] PMMP  5e-5  u=5:  DEAD — collapsed ep21→22 (loss 3.4→6.9), val 0.1%  ckpt ep25 useless  260555_6
#   [sub 7] PMMP  2.5e-6 u=2:  RUNNING   277380_7
#   [sub 8] PMMP  2.5e-6 u=5:  RUNNING   277380_8
#   [sub 9] PMMP  5e-6   u=2:  RUNNING   277380_9
#   [sub 10] PMMP 5e-6   u=5:  RUNNING   277380_10
#####

args = TrainArgs{Float32}()

# Load configuration
path_to_db, imagenet_path, _ = load_imagenet_config()

experiment_name = "alpha-pmmp-v1"

single_run_routine = single_run_routine_classifier

variables = [:α, :initial_u_value]

batch = [
    (1f-6, 2f0),   # sub 1
    (1f-5, 2f0),   # sub 2
    (5f-5, 2f0),   # sub 3
    (1f-6, 5f0),   # sub 4
    (1f-5, 5f0),   # sub 5
    (5f-5, 5f0),   # sub 6
    (2.5f-6, 2f0), # sub 7
    (2.5f-6, 5f0), # sub 8
    (5f-6, 2f0),   # sub 9
    (5f-6, 5f0),   # sub 10
]

# Fixed arguments for all runs
args.optimization_procedure = PMMP_procedure
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

# PMMP-specific
args.initial_p_value = 1f0

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
