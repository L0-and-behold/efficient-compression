"""
    CIFAR-C.jl

Tests the robustness of models trained on CIFAR using tests on the CIFAR-C dataset, cf. https://github.com/hendrycks/robustness
"""

using Pkg; Pkg.activate("."); using Revise

begin
    using Lux
    using MLDatasets: CIFAR10
    using Random
    using Statistics: mean
    using JSON3
    using CSV, DataFrames
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

function total_loss(tstate, dataset, loss_fun)
    return compute_loss_over_batches(tstate, tstate.parameters, dataset, Float32, loss_fun)
end

model_path = "./src/DatasetsModels/CIFAR-C/tested_models/sweep/1/"

train_set, validation_set, test_set = CIFAR_data(args.train_batch_size, args.dev; seed=1234);

store = Dict{String, Dict{String, Any}}()

for entry in readdir(model_path)
# entry = readdir(model_path)[1]
    subfolder = joinpath(model_path, entry)
    m = match(r"_(\d+)-\d+", subfolder) # matches digits after _ and before -digits
    subfolder_index = m.captures[1]
    isdir(subfolder) || error("Not a directory: $subfolder")

    # extract name, alpha and seed from csv file
    csv_files = filter(f -> endswith(f, ".csv"), readdir(subfolder))
    isempty(csv_files) && error("No CSV file found in: $subfolder") # skip if no csv found
    csvpath = joinpath(subfolder, csv_files[1])
    df = CSV.read(csvpath, DataFrame)  # first row = header (names), second row = values
    row = df[1, :] # read the single data row
    if row.optimization_procedure == "RL1_procedure" && row.α == 0f0 && row.β == 0f0 && row.initial_p_value == 0f0
        method_name = "vanilla"
    else
        method_name = row.optimization_procedure
    end
    storage_name = method_name * "_seed-" * string(row.seed) * "_alpha-" * string(row.α) * "_index-" * subfolder_index
    sub_store = get!(store, storage_name, Dict{String, Any}())
    sub_store["method_name"] = method_name
    sub_store["seed"] = row.seed
    sub_store["alpha"] = row.α
    println(method_name, ", seed = ", row.seed, ", α = ", row.α, ", index = ", subfolder_index)

    artifacts_subfolder = subfolder * "/artifacts/"
    run_subfolder_name = readdir(artifacts_subfolder)[1]
    run_subfolder = artifacts_subfolder * run_subfolder_name
    
    tstate_cpu, model, rng = load_train_state(run_subfolder * "/train_state.bson");
    tstate = to_gpu(tstate_cpu);
    loss_fun = initialize_RL1_loss(tstate, args, logitcrossentropy)

    clean_acc = accuracy(tstate,test_set)
    clean_loss = total_loss(tstate, test_set, loss_fun)

    sub_store["clean_acc"] = clean_acc
    sub_store["clean_loss"] = clean_loss

    CAs = []
    CEs = []
    for corruption_type in corruption_types
        println(corruption_type)
        corrupted_test_set = CIFAR_C_data(corruption_type)

        corrupted_acc = accuracy(tstate, corrupted_test_set)
        push!(CAs, corrupted_acc)
        println(corrupted_acc, " = corrupted_acc")

        corrupted_loss = total_loss(tstate, corrupted_test_set, loss_fun)
        push!(CEs, corrupted_loss)
        println(corrupted_loss, " = corrupted_loss")
        println()
    end

    sub_store["CAs"] = CAs
    sub_store["CEs"] = CEs

    println()
end

open("./src/DatasetsModels/CIFAR-C/tested_models/sweep/1/store.json", "w") do io
    JSON3.pretty(io, store)
end

















function compute_metrics(stored_data)
    mCA = mean(CAs)
    clean_acc
    mCA_vanilla = mean(CAs_vanilla)
    clean_acc_vanilla
    Diff_mCA = mCA - mCA_vanilla
    mCE = mean(CEs)
    clean_loss
    mCE_vanilla = mean(CEs_vanilla)
    clean_loss_vanilla
    Diff_mCE = mCE - mCE_vanilla 

    NCAs = CAs ./ CAs_vanilla
    mNCA = mean(NCAs)
    NCEs = CEs ./ CEs_vanilla
    mNCE = mean(NCEs)

    Relative_CAs = CAs .- clean_acc
    Relative_mCA = mean(Relative_CAs)
    Relative_CAs_vanilla = CAs_vanilla .- clean_acc_vanilla
    Relative_mCA_vanilla = mean(Relative_CAs_vanilla)
    Relative_NCAs = Relative_CAs ./ Relative_CAs_vanilla
    Relative_mNCA = mean(Relative_NCAs)
    Relative_CEs = CEs .- clean_loss
    Relative_mCE = mean(Relative_CEs)
    Relative_CEs_vanilla = CEs_vanilla .- clean_loss_vanilla
    Relative_NCEs = Relative_CEs ./ Relative_CEs_vanilla
    Relative_mNCE = mean(Relative_NCEs)
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