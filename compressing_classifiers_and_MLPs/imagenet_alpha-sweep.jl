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
# Results summary — vanilla ep25=60.4%, ep28=60.7%, ep29=60.6%, ep30=63.9%, ep31=64.5%, ep32=60.3%
#   [sub  1] RL1   2e-7:  210054_1 OOM → 216135  ep28 val=59.8% (Δ-0.9pp)  CR@ep25=51.1%
#   [sub  2] RL1   4e-7:  210054_2               ep31 val=63.4% (Δ-1.1pp)  CR@ep30=53.5%
#   [sub  3] RL1   8e-7:  210054_3 OOM → 216136  ep29 val=57.0% (Δ-3.6pp)  CR@ep25=60.9%
#   [sub  4] RL1   1e-6:  210054_4          ep31 val=59.5% (Δ-5.0pp)  CR@ep30=67.4%
#   [sub  5] RL1   4e-6:  210054_5 STOPPED  ep14 val=46.3% (Δ-12.5pp) CR@ep10=92.0%  — 12.5pp gap after 14ep, unrecoverable
#   [sub  6] DRR   1e-8:  216130_6
#   [sub  7] DRR   1e-7:  210054_7 OOM → 216137  ep31 val=62.2% (Δ-2.3pp)  CR@ep30=58.1%
#   [sub  8] DRR   2e-7:  210054_8 OOM → 216130_8
#   [sub  9] DRR   4e-7:  210054_9          ep32 val=60.4% (Δ+0.1pp)  CR@ep30=83.5%
#   [sub 10] PMMP  1e-10: 216130_10
#   [sub 11] PMMP  1e-9:  216130_11
#   [sub 12] PMMP  1e-8:  216130_12
#   [sub 13] PMMP  1e-7:  216130_13
#   [sub 14] RL1   9e-7:  216130_14
#####

args = TrainArgs{Float32}()

# Load configuration
path_to_db, imagenet_path, _ = load_imagenet_config()

experiment_name = "alpha-sweep-v1"

single_run_routine = single_run_routine_classifier

variables = [:optimization_procedure, :α]

batch = [
    (RL1_procedure, 2f-7),   # sub 1
    (RL1_procedure, 4f-7),   # sub 2
    (RL1_procedure, 8f-7),   # sub 3
    (RL1_procedure, 1f-6),   # sub 4
    (RL1_procedure, 4f-6),   # sub 5
    (DRR_procedure, 1f-8),   # sub 6
    (DRR_procedure, 1f-7),   # sub 7
    (DRR_procedure, 2f-7),   # sub 8
    (DRR_procedure, 4f-7),   # sub 9
    (PMMP_procedure, 1f-10), # sub 10
    (PMMP_procedure, 1f-9),  # sub 11
    (PMMP_procedure, 1f-8),  # sub 12
    (PMMP_procedure, 1f-7),  # sub 13
    (RL1_procedure,  9f-7),  # sub 14
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
