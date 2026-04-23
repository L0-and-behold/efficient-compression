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
#
# Results summary — vanilla ep36=62.4%, ep50=63.2%, ep54=66.7%, ep58=67.5%, ep60=66.8%, ep70=67.2%, ep71=70.5%, ep80=69.7%, ep81=71.1%, ep88=71.8%
#   [sub  1] RL1   2e-7:  STOPPED   ep62 val=65.9% (Δ-1.3pp)  CR@ep60=45.0%  — low CR    210054_1 OOM → 216135 OOM → 228421 FAIL → 244942 OOM → 260546
#   [sub  2] RL1   4e-7:  FINISHED  val=73.29 (Δ-1.1pp) CR@pruning=63.5               210054_2 OOM → 260021 FAIL → 260652(1FT) → 273245 FAIL → 276293 FAIL → 278630 FAIL → 278770 stopped
#   [sub  3] RL1   8e-7:  FINISHED   val=73.31 (Δ-1.1pp)  CR@pruning=61.1%             210054_3 OOM → 216136 OOM → 228422 FAIL → 244943 OOM → 260023 OOM → 277431 FAIL → 278771 OOM
#   [sub  4] RL1   1e-6:  FINISHED   val=72.16% (Δ±0.0pp)  CR@FT-epoch-5=74.3%    210054_4 OOM → 260022 OOM → 260547 FAIL → 260653(1FT) → 273246 FAIL → 276294 FAIL → 278631 stopped
#   [sub  5] RL1   4e-6:  STOPPED  -12pp val acc gap @ ep14                              210054_5
#   [sub  6] DRR   1e-8:  STOPPED  ep79 val=68.7% (Δ-2.3pp)  CR@ep75=41.7%  — low CR    216130_6
#   [sub  7] DRR   1e-7:  FINISHED  val=71.06% (Δ-1.1pp)  CR@ep90=64.9    210054_7 OOM → 216137 OOM(FT) → 260548 FAIL → 260654(1FT) → 273247 FAIL → 276295 FAIL → 278632 stopped
#   [sub  8] DRR   2e-7:  STOPPED  ckpt@ep25 (too slow & we already have good drr values)  210054_8 OOM → 216130_8 OOM → 228423 OOM → 260236 OOM → 260656 OOM → 278772 stopped
#   [sub  9] DRR   4e-7:  FINISHED  eval=73.418% (Δ+1.2pp)  CR@ep90=87.5%     210054_9 OOM(FT) → 260235 FAIL → 260245(1FT) → 260549 FAIL → 260655(1FT) → 273248 FAIL → 276296 FAIL → 278633 stopped
#   [sub 10] PMMP  1e-10: STOPPED  ep30 val=63.3% (Δ-0.6pp)  CR@ep30=40.9%  — low CR    216130_10
#   [sub 11] PMMP  1e-9:  STOPPED  ep30 val=65.3% (Δ+1.3pp)  CR@ep30=40.4%  — low CR    216130_11
#   [sub 12] PMMP  1e-8:  DEAD — loss collapsed ep19→ep20 (2.578→6.907), val 0.1% onward 216130_12 OOM → 228424
#   [sub 13] PMMP  1e-7:  STOPPED  ep71 val=69.3% (Δ+2.5pp)  CR@ep70=36.6%  — low CR    216130_13
#   [sub 14] RL1   9e-7:  FINISHED  ep90+10: val=72.1% (Δ-0.06pp)  CR=68.0% ie. 3.12x     216130_14
#   [sub 15] PMMP  1e-6:  STOPPED  ep27 val=61.0% (Δ+2.8pp)  CR@ep25=38.3%  — low CR    244945 OOM → 260550
#   [sub 16] PMMP  1e-5:  DEAD — loss collapsed ~ep20, val 0.1%                          244946
#   [sub 17] PMMP  2e-8:  STOPPED  ep17 val=58.7% (Δ+0.3pp)  CR@ep15=39.2%  — low CR    260240                           
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
    (PMMP_procedure, 1f-6),  # sub 15
    (PMMP_procedure, 1f-5),  # sub 16
    (PMMP_procedure, 2f-8),  # sub 17
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
