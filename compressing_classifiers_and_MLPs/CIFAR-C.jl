"""
    CIFAR-C.jl

Tests the robustness of models trained on CIFAR using tests on the CIFAR-C dataset, cf. https://github.com/hendrycks/robustness
"""

using Pkg; Pkg.activate("."); using Revise

begin
    using Lux
    using MLDatasets: CIFAR10
    using Random
end
using CompressingClassifiersMLPs

using CompressingClassifiersMLPs.TrainingArguments: TrainArgs
using CompressingClassifiersMLPs.TrainingTools: save_train_state, load_train_state, to_cpu, to_gpu
using CompressingClassifiersMLPs.OptimizationProcedures: PMMP_procedure, RL1_procedure, DRR_procedure, layerwise_procedure, Lenet_MLP, Lenet_5, Lenet_5_Caffe, VGG, logitcrossentropy, logitcrossentropy_ls, generate_tstate, accuracy, initialize_DRR_loss, initialize_RL1_loss, initialize_PMMP_loss, testmode_states, compute_loss_over_batches
using CompressingClassifiersMLPs.DatasetsModels: MNIST_data, CIFAR_data, CIFAR_C_data, corruption_types
using CompressingClassifiersMLPs.BatchRun: do_batch_run, get_sub_batch, single_run_routine_classifier, single_run_routine_teacherstudent

using CompressingClassifiersMLPs.Checkpointer

begin
    args = TrainArgs{Float32}()
    args.architecture = VGG
    args.dataset = CIFAR_data
    args.train_batch_size = 500
    args.smoothing_window = 20
    args.min_epochs = 30
    args.max_epochs = 300
    args.finetuning_min_epochs = 10
    args.finetuning_max_epochs = 50
    args.train_set_size = 45000
    args.val_set_size = 5000
    args.val_batch_size = 5000
    args.test_set_size = 10000
    args.test_batch_size = 10000
    args.noise = 0f0
    args.prune_window = 10
    args.shrinking_from_deviation_of = 1e-2
    args.gauss_loss = false
    args.dev = Lux.gpu_device()
end


# single_run_routine = single_run_routine_classifier
# Directory for saving results
# path_to_db = joinpath(pwd(), "experiment-results")
# experiment_name = "CIFAR-C-experiment"
# train_set, validation_set, test_set = CIFAR_data(args.train_batch_size, args.dev; seed=123);
# loss_fctn = args.label_smoothing ? logitcrossentropy_ls : logitcrossentropy
# model = VGG(dropout=0.0f0);
# initial_parameter_count = Lux.parameterlength(model)
# 15253578 for VGG-16-512
# tstate = generate_tstate(model, model_seed, args.optimizer(args.lr); dev=args.dev);
# checkpoint = CheckpointManager(
#         args.use_checkpoints,
#         CheckpointMetadata(path=joinpath(path_to_db, experiment_name, "checkpoints")),
#         CheckpointContent(args=args)
#     )
# @time tstate, logs, loss_fun, checkpoint = RL1_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args, checkpoint);
# 


tstate_cpu, model, rng = load_train_state("./src/DatasetsModels/CIFAR-C/tested_models/no_shrinking_val/CIFAR_5_val_acc_1-5/artifacts/run-6fa7682b/train_state.bson");
tstate = to_gpu(tstate_cpu);

tstate_cpu_vanilla, model_vanilla, rng_vanilla = load_train_state("./src/DatasetsModels/CIFAR-C/tested_models/no_shrinking_val/CIFAR_5_val_acc_2-5/artifacts/run-361a3d4e/train_state.bson");
tstate_vanilla = to_gpu(tstate_cpu_vanilla);

train_set, validation_set, test_set = CIFAR_data(args.train_batch_size, args.dev; seed=1234);
test_set_corrupted = CIFAR_C_data(corruption_types[1]);

base_acc = accuracy(tstate,test_set) # 0.8384f0
corrupted_acc_1 = accuracy(tstate,test_set_corrupted) # 0.81872f0
dif = base_acc - corrupted_acc_1 # 0.019680023f0

base_acc_vanilla = accuracy(tstate_vanilla,test_set) # 0.8335f0
corrupted_acc_1_vanilla = accuracy(tstate_vanilla,test_set_corrupted) # 0.80392f0
diff = base_acc_vanilla - corrupted_acc_1_vanilla # 0.029580057f0

loss_fun = initialize_RL1_loss(tstate, args, logitcrossentropy)
function total_loss(tstate, dataset)
    return compute_loss_over_batches(tstate, tstate.parameters, dataset, Float32, loss_fun)
end

CAs = []
CAs_vanilla = []
CEs = []
CEs_vanilla = []
for corruption_type in corruption_types
    corrupted_test_set = CIFAR_C_data(corruption_type)

    corrupted_acc = accuracy(tstate, corrupted_test_set)
    push!(CAs, corrupted_acc)
    corrupted_acc_vanilla = accuracy(tstate_vanilla, corrupted_test_set)
    push!(CAs_vanilla, corrupted_acc_vanilla)

    corrupted_loss = total_loss(tstate, corrupted_test_set)
    push!(CEs, corrupted_loss)
    corrupted_loss_vanilla = total_loss(tstate_vanilla, corrupted_test_set)
    push!(CEs_vanilla, corrupted_loss_vanilla)
end
mCA = mean(CAs)
clean_acc
mCA_vanilla = mean(CAs_vanilla)
clean_acc_vanilla
Diff_mCA = mCA - mCA_vanilla
mCE = mean(CEs)
clean_err
mCE_vanilla = mean(CEs_vanilla)
clean_err_vanilla
Diff_mCE = mCE - mCE_vanilla 

NCAs = CAs ./ CAs_vanilla
mNCA = mean(NCAs)
NCEs = CEs ./ CEs_vanilla
mNCE = mean(NCEs)

Relative_CAs = CAs .- clean_acc
Relative_mCA = mean(Relative_CAs)
Relative_CAs_vanilla = CAs_vanilla .- clean_acc_vanilla
Relative_NCAs = Relative_CAs ./ Relative_CAs_vanilla
Relative_mNCA = mean(Relative_NCAs)
Relative_CEs = CEs .- clean_err
Relative_mCE = mean(Relative_CEs)
Relative_CEs_vanilla = CEs_vanilla .- clean_err_vanilla
Relative_NCEs = Relative_CEs ./ Relative_CEs_vanilla
Relative_mNCE = mean(Relative_NCEs)