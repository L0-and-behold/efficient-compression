"""
    run_ECE_benchmark.jl

Expected Calibration Error (ECE) benchmark for CIFAR-10 models.
Evaluates calibration of baseline (vanilla) and compressed models (RL1, DRR, PMMP)
on the clean CIFAR-10 test set. Produces:
  - One reliability diagram PNG per method
  - JSON file with numerical ECE/MCE results
  - Summary table printed to stdout

Reference: Guo et al. 2017, "On Calibration of Modern Neural Networks"
"""

using Pkg; Pkg.activate("."); using Revise

begin
    using Lux
    using MLDatasets: CIFAR10
    using Random
    using Statistics: mean
    using JSON3
    using CSV, DataFrames
    using Plots
    using BSON
    using Optimisers
end
using CompressingClassifiersMLPs

using CompressingClassifiersMLPs.TrainingArguments: TrainArgs
using CompressingClassifiersMLPs.TrainingTools: save_train_state, load_train_state, to_cpu, to_gpu
using CompressingClassifiersMLPs.OptimizationProcedures: PMMP_procedure, RL1_procedure, DRR_procedure, Lenet_MLP, Lenet_5, Lenet_5_Caffe, VGG, logitcrossentropy, logitcrossentropy_ls, generate_tstate, accuracy, initialize_DRR_loss, initialize_RL1_loss, initialize_PMMP_loss, testmode_states, collect_logits_and_labels, expected_calibration_error, reliability_diagram_data
using CompressingClassifiersMLPs.DatasetsModels: MNIST_data, CIFAR_data
using CompressingClassifiersMLPs.BatchRun: do_batch_run, get_sub_batch, single_run_routine_classifier

using CompressingClassifiersMLPs.Checkpointer

# ─── Configuration ───────────────────────────────────────────────────────────

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

N_BINS = 15
model_path = "./src/DatasetsModels/CIFAR-C/tested_models/sweep/1/"
output_dir = "./experiment-results/ece_benchmark/"
mkpath(output_dir)

# ─── Load clean CIFAR-10 test set ────────────────────────────────────────────

train_set, validation_set, test_set = CIFAR_data(args.train_batch_size, args.dev; seed=1234);

# ─── Evaluate models ─────────────────────────────────────────────────────────

results = Dict{String, Any}()

for entry in readdir(model_path)
    subfolder = joinpath(model_path, entry)
    isdir(subfolder) || continue

    m = match(r"_(\d+)-\d+", subfolder)
    isnothing(m) && continue
    subfolder_index = m.captures[1]

    # Extract method name, alpha, seed from CSV
    csv_files = filter(f -> endswith(f, ".csv"), readdir(subfolder))
    isempty(csv_files) && continue
    csvpath = joinpath(subfolder, csv_files[1])
    df = CSV.read(csvpath, DataFrame)
    row = df[1, :]

    if row.optimization_procedure == "RL1_procedure" && row.α == 0f0 && row.β == 0f0 && row.initial_p_value == 0f0
        method_name = "vanilla"
    else
        method_name = row.optimization_procedure
    end
    storage_name = method_name * "_seed-" * string(row.seed) * "_alpha-" * string(row.α) * "_index-" * subfolder_index

    println("Loading: $storage_name")

    # Load model
    artifacts_subfolder = joinpath(subfolder, "artifacts")
    run_subfolder_name = readdir(artifacts_subfolder)[1]
    run_subfolder = joinpath(artifacts_subfolder, run_subfolder_name)

    tstate_cpu, model, rng = load_train_state(joinpath(run_subfolder, "train_state.bson"))
    tstate = to_gpu(tstate_cpu)

    # Collect logits and compute ECE
    logits, labels = collect_logits_and_labels(tstate, test_set)
    rd = reliability_diagram_data(logits, labels; n_bins=N_BINS)

    # Store results
    results[storage_name] = Dict(
        "method_name" => method_name,
        "seed" => row.seed,
        "alpha" => row.α,
        "ece" => rd.ece,
        "mce" => rd.mce,
        "accuracy" => accuracy(tstate, test_set),
        "bin_accuracies" => replace(rd.bin_accuracies, NaN32 => -1f0),
        "bin_confidences" => replace(rd.bin_confidences, NaN32 => -1f0),
        "bin_counts" => rd.bin_counts,
    )

    println("  ECE = $(round(rd.ece; digits=4)), MCE = $(round(rd.mce; digits=4))")
    println()
end

# ─── Save numerical results ──────────────────────────────────────────────────

open(joinpath(output_dir, "ece_results.json"), "w") do io
    JSON3.pretty(io, results)
end
println("Results saved to $(joinpath(output_dir, "ece_results.json"))")

# ─── Generate reliability diagram PNGs (one per method) ──────────────────────

function plot_reliability_diagram(method_name, method_results; output_dir=output_dir, n_bins=N_BINS)
    bin_edges = range(0.0, 1.0, length=n_bins + 1)
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in 1:n_bins]
    bin_width = 1.0 / n_bins

    # Collect all runs for this method
    all_accuracies = [r["bin_accuracies"] for r in method_results]
    all_confidences = [r["bin_confidences"] for r in method_results]
    all_counts = [r["bin_counts"] for r in method_results]
    eces = [r["ece"] for r in method_results]

    # Average over runs (ignoring empty bins marked as -1)
    avg_acc = fill(NaN, n_bins)
    avg_conf = fill(NaN, n_bins)
    for b in 1:n_bins
        valid_accs = [a[b] for a in all_accuracies if a[b] >= 0]
        valid_confs = [c[b] for c in all_confidences if c[b] >= 0]
        if !isempty(valid_accs)
            avg_acc[b] = mean(valid_accs)
            avg_conf[b] = mean(valid_confs)
        end
    end

    mean_ece = mean(eces)

    # Plot
    p = plot(
        size=(600, 500),
        title="Reliability Diagram: $method_name\n(ECE = $(round(mean_ece; digits=4)))",
        xlabel="Confidence",
        ylabel="Accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
        legend=:topleft,
        grid=true,
        framestyle=:box,
    )

    # Perfect calibration diagonal
    plot!(p, [0, 1], [0, 1], linestyle=:dash, color=:gray, linewidth=1.5, label="Perfect calibration")

    # Bar chart of actual accuracy per bin
    non_nan = .!isnan.(avg_acc)
    bar!(p, bin_centers[non_nan], avg_acc[non_nan],
        bar_width=bin_width * 0.9,
        color=:steelblue,
        alpha=0.7,
        label="Outputs",
    )

    # Gap (overconfidence) shading
    for b in 1:n_bins
        if !isnan(avg_acc[b]) && !isnan(avg_conf[b])
            gap_color = avg_conf[b] > avg_acc[b] ? :salmon : :lightgreen
            bar_lo = min(avg_acc[b], avg_conf[b])
            bar_hi = max(avg_acc[b], avg_conf[b])
            if bar_hi - bar_lo > 1e-4
                plot!(p,
                    Shape([
                        bin_centers[b] - bin_width*0.45, bin_centers[b] + bin_width*0.45,
                        bin_centers[b] + bin_width*0.45, bin_centers[b] - bin_width*0.45
                    ], [bar_lo, bar_lo, bar_hi, bar_hi]),
                    fillcolor=gap_color, fillalpha=0.5, linecolor=:transparent, label=(b == findfirst(!isnan, avg_acc) ? "Gap" : ""),
                )
            end
        end
    end

    savefig(p, joinpath(output_dir, "reliability_diagram_$(method_name).png"))
    println("Saved: reliability_diagram_$(method_name).png")
end

# Group results by method
method_groups = Dict{String, Vector{Dict}}()
for (name, r) in results
    method = r["method_name"]
    if !haskey(method_groups, method)
        method_groups[method] = []
    end
    push!(method_groups[method], r)
end

# Generate one plot per method
for (method_name, method_results) in method_groups
    plot_reliability_diagram(method_name, method_results)
end

# ─── Print summary table ─────────────────────────────────────────────────────

println("\n" * "="^70)
println("ECE Benchmark Summary (CIFAR-10 Clean Test Set, $N_BINS bins)")
println("="^70)
println(rpad("Method", 20), rpad("Seed", 8), rpad("α", 10), rpad("Acc", 10), rpad("ECE", 10), "MCE")
println("-"^70)
for (name, r) in sort(collect(results); by=x -> x.second["method_name"])
    println(
        rpad(r["method_name"], 20),
        rpad(string(r["seed"]), 8),
        rpad(string(round(r["alpha"]; digits=4)), 10),
        rpad(string(round(r["accuracy"]; digits=4)), 10),
        rpad(string(round(r["ece"]; digits=4)), 10),
        string(round(r["mce"]; digits=4)),
    )
end
println("="^70)
