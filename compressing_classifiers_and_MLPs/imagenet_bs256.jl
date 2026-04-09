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

path_to_db, imagenet_path, _ = load_imagenet_config()

experiment_name = "vanilla-bs256"

single_run_routine = single_run_routine_classifier

variables = [:optimization_procedure, :lr, :α, :initial_p_value, :initial_u_value]

batch = [
    (RL1_procedure,  0.1f0, 0f0, 0f0, 0f0),   # vanilla: memory + speed baseline
    (PMMP_procedure, 0.1f0, 1f-6, 1f0, 1f0),  # PMMP: memory stress test at bs=256
]

args.seed = 1
args.architecture = resnet50
args.dataset = imagenet_data_function()
args.train_set_size = 1_281_024
args.val_set_size = 50000
args.test_set_size = 50000
args.train_batch_size = 256
args.val_batch_size = 128

args.smoothing_window = 1000
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

args.min_epochs            = 90
args.max_epochs            = 90
args.finetuning_min_epochs = 0
args.finetuning_max_epochs = 0

args.tamade_calibration_batches = 200
args.save_pre_pruning_model = true
args.skip_precompilation = true
args.debug = false
args.use_checkpoints = true
args.checkpoint_frequency = 10
args.cr_report_window = 10

args.label_smoothing = true
args.ρ = 8f-6
args.finetuning_min_epochs = 10
args.finetuning_max_epochs = 10

args.optimizer = lr -> Optimisers.OptimiserChain(Optimisers.Momentum(lr, 0.9f0))
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

experiment_name, batch = get_sub_batch(experiment_name, batch)
args.resume_checkpoint_id = parse_resume_checkpoint()

do_batch_run(path_to_db, experiment_name, single_run_routine, args, variables, batch; break_if_one_run_errors=true)
