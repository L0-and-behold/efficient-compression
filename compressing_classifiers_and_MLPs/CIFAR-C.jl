"""
    CIFAR-C.jl

Tests the robustness of models trained on CIFAR using tests on the CIFAR-C dataset, cf. https://github.com/hendrycks/robustness
"""

using Pkg; Pkg.activate("."); using Revise

begin
    using Lux
    using MLDatasets: CIFAR10
    using Random
    using Statistics: mean, std
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

open(model_path * "store.json", "w") do io
    JSON3.pretty(io, store)
end



seeds = [5*n for n in 0:11]

results = Dict{String, Dict{String, Any}}()
for seed in seeds
    subdict = Dict(k => v for (k, v) in store if v["seed"] == seed)
    vanilla = [v for (k, v) in subdict  if v["method_name"] == "vanilla"][1]
    for method in values(subdict)
        if method["method_name"] != "vanilla"
            sub_results = get!(results, method["method_name"] * "_seed-" * string(seed), Dict{String, Any}())
            sub_results["seed"] = seed
            sub_results["method_name"] = method["method_name"]

            CAs = method["CAs"]
            CEs = method["CEs"]
            CAs_vanilla = vanilla["CAs"]
            CEs_vanilla = vanilla["CEs"]
            clean_acc = method["clean_acc"]
            clean_loss = method["clean_loss"]
            clean_acc_vanilla = vanilla["clean_acc"]
            clean_loss_vanilla = vanilla["clean_loss"]

            mCA = mean(CAs)
            mCA_vanilla = mean(CAs_vanilla)
            mCE = mean(CEs)
            mCE_vanilla = mean(CEs_vanilla)

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

            sub_results["clean_acc"] = clean_acc
            sub_results["clean_loss"] = clean_loss
            sub_results["clean_acc_vanilla"] = clean_acc_vanilla
            sub_results["clean_loss_vanilla"] = clean_loss_vanilla
            sub_results["mCA"] = mCA
            sub_results["mCA_vanilla"] = mCA_vanilla
            sub_results["mCE"] = mCE
            sub_results["mCE_vanilla"] = mCE_vanilla
            sub_results["mNCA"] = mNCA
            sub_results["mNCE"] = mNCE
            sub_results["Relative_mCA"] = Relative_mCA
            sub_results["Relative_mCA_vanilla"] = Relative_mCA_vanilla
            sub_results["Relative_mNCA"] = Relative_mNCA
            sub_results["Relative_mCE"] = Relative_mCE
            sub_results["Relative_mNCE"] = Relative_mNCE
        end
    end
end

open(model_path * "results.json", "w") do io
    JSON3.pretty(io, results)
end

# compute average and std deviation over seeds
final_results = Dict{String, Dict{String, Any}}()
method_names = ["DRR_procedure", "RL1_procedure", "PMMP_procedure"]
for method_name in method_names
    same_method_subdict = Dict(k => v for (k, v) in results if v["method_name"] == method_name)

    same_method_subdict_values = collect(values(same_method_subdict))
    same_method_subdict_keys = keys(first(same_method_subdict_values))

    averages = get!(final_results, method_name, Dict{String, Any}())

    for k in same_method_subdict_keys
        vals = [sub[k] for sub in same_method_subdict_values]
        if vals[1] isa AbstractString
            averages[k] = vals[1]  # store the (always equal) string once
        else
            averages[k] = Dict("mean"=>mean(vals),"std" => std(vals))   # (mean, std) tuple
        end
    end
end

open(model_path * "final_results.json", "w") do io
    JSON3.pretty(io, final_results)
end