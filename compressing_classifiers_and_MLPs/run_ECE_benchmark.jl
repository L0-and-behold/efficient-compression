"""
    run_ECE_benchmark.jl

Expected Calibration Error (ECE) benchmark for CIFAR-10 models.
Evaluates calibration of baseline (vanilla) and compressed models (RL1, DRR, PMMP)
on the clean CIFAR-10 test set. Produces:
  - JSON file with numerical ECE/MCE results
  - .out report with summary table

Reference: Guo et al. 2017, "On Calibration of Modern Neural Networks"
"""

using Pkg; Pkg.activate("."); using Revise

begin
    using Lux
    using MLDatasets: CIFAR10
    using Random
    using Statistics: mean, std
    using JSON3
    using CSV, DataFrames
    using BSON
    using Optimisers
    using Dates
end
using CompressingClassifiersMLPs

using CompressingClassifiersMLPs.TrainingArguments: TrainArgs
using CompressingClassifiersMLPs.TrainingTools: save_train_state, load_train_state, to_cpu, to_gpu
using CompressingClassifiersMLPs.OptimizationProcedures: PMMP_procedure, RL1_procedure, DRR_procedure, Lenet_MLP, Lenet_5, Lenet_5_Caffe, VGG, logitcrossentropy, logitcrossentropy_ls, generate_tstate, accuracy, initialize_DRR_loss, initialize_RL1_loss, initialize_PMMP_loss, testmode_states, collect_logits_and_labels, expected_calibration_error, reliability_diagram_data
using CompressingClassifiersMLPs.DatasetsModels: MNIST_data, CIFAR_data
using CompressingClassifiersMLPs.BatchRun: do_batch_run, get_sub_batch, single_run_routine_classifier

using CompressingClassifiersMLPs.Checkpointer
using CompressingClassifiersMLPs.Config: load_ece_config

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
MIN_ACCURACY = 0.70  # exclude (collapsed) runs with test accuracy below this threshold
model_path = load_ece_config()
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

# ─── Group results by method (dropping seeds where any method failed) ─────────

# Find seeds where ANY method's accuracy is below threshold
all_seeds_in_results = Set(r["seed"] for (_, r) in results)
excluded_seeds = Set{Any}()
for s in all_seeds_in_results
    for (_, r) in results
        if r["seed"] == s && r["accuracy"] < MIN_ACCURACY
            push!(excluded_seeds, s)
            break
        end
    end
end
if !isempty(excluded_seeds)
    println("Excluded seeds (some method has acc < $MIN_ACCURACY): $excluded_seeds")
    for (name, r) in results
        if r["seed"] in excluded_seeds && r["accuracy"] < MIN_ACCURACY
            println("  → $(name): accuracy=$(round(r["accuracy"]; digits=4))")
        end
    end
    println()
end

method_groups = Dict{String, Vector{Dict}}()
for (name, r) in results
    if r["seed"] in excluded_seeds
        continue
    end
    method = r["method_name"]
    if !haskey(method_groups, method)
        method_groups[method] = []
    end
    push!(method_groups[method], r)
end

# ─── Generate .out report with paired-difference ΔECE ± SE ───────────────────

function generate_report(results, method_groups; output_dir=output_dir, n_bins=N_BINS)
    report_path = joinpath(output_dir, "ece_report.out")

    # Build seed → ECE lookup per method
    method_seed_ece = Dict{String, Dict{Any, Float64}}()
    method_seed_acc = Dict{String, Dict{Any, Float64}}()
    for (method, runs) in method_groups
        method_seed_ece[method] = Dict(r["seed"] => r["ece"] for r in runs)
        method_seed_acc[method] = Dict(r["seed"] => r["accuracy"] for r in runs)
    end

    # Paired seeds (intersection of all methods)
    all_seeds = sort(collect(keys(method_seed_ece["vanilla"])))

    # Compute paired differences: δ(s) = ECE_method(s) - ECE_vanilla(s)
    # and paired relative differences: δ_rel(s) = δ(s) / ECE_vanilla(s)
    paired_deltas = Dict{String, Vector{Float64}}()
    paired_rel_deltas = Dict{String, Vector{Float64}}()
    for method in ["RL1_procedure", "DRR_procedure", "PMMP_procedure"]
        haskey(method_seed_ece, method) || continue
        deltas = Float64[]
        rel_deltas = Float64[]
        for s in all_seeds
            if haskey(method_seed_ece[method], s)
                d = method_seed_ece[method][s] - method_seed_ece["vanilla"][s]
                push!(deltas, d)
                push!(rel_deltas, d / method_seed_ece["vanilla"][s])
            end
        end
        paired_deltas[method] = deltas
        paired_rel_deltas[method] = rel_deltas
    end

    # Per-method aggregate stats
    method_stats = Dict{String, NamedTuple}()
    for (method, runs) in method_groups
        eces = [r["ece"] for r in runs]
        mces = [r["mce"] for r in runs]
        accs = [r["accuracy"] for r in runs]
        method_stats[method] = (
            mean_ece = mean(eces), std_ece = length(eces) > 1 ? std(eces) : 0.0,
            mean_mce = mean(mces), mean_acc = mean(accs), n_runs = length(runs),
        )
    end

    open(report_path, "w") do io
        println(io, "=" ^ 100)
        println(io, "  ECE Benchmark Report — CIFAR-10 Clean Test Set ($n_bins bins)")
        println(io, "  Generated: $(Dates.now())")
        println(io, "  Seeds: $(all_seeds)")
        println(io, "  Excluded seeds (any method acc < $(MIN_ACCURACY)): $(isempty(excluded_seeds) ? "none" : excluded_seeds)")
        println(io, "=" ^ 100)
        println(io)

        # ── Main summary table with paired ΔECE ──
        println(io, "─" ^ 100)
        println(io, rpad("Method", 18), rpad("n", 4), rpad("Acc", 12),
                    rpad("ECE", 16), rpad("ΔECE (paired)", 22),
                    rpad("ΔECE% (paired)", 22), "MCE")
        println(io, "─" ^ 100)

        for method in ["vanilla", "RL1_procedure", "DRR_procedure", "PMMP_procedure"]
            haskey(method_stats, method) || continue
            s = method_stats[method]

            if method == "vanilla"
                delta_str = "— (baseline)"
                rel_str = "—"
            else
                deltas = paired_deltas[method]
                n = length(deltas)
                mean_delta = mean(deltas)
                se_delta = n > 1 ? std(deltas) / sqrt(n) : 0.0
                sign_str = mean_delta >= 0 ? "+" : ""
                delta_str = "$(sign_str)$(round(mean_delta; digits=4)) ± $(round(se_delta; digits=4))"

                rel_deltas = paired_rel_deltas[method]
                mean_rel = mean(rel_deltas) * 100
                se_rel = (n > 1 ? std(rel_deltas) / sqrt(n) : 0.0) * 100
                rel_sign = mean_rel >= 0 ? "+" : ""
                rel_str = "$(rel_sign)$(round(mean_rel; digits=1))% ± $(round(se_rel; digits=1))%"
            end

            println(io,
                rpad(method, 18),
                rpad(string(s.n_runs), 4),
                rpad("$(round(s.mean_acc; digits=4))", 12),
                rpad("$(round(s.mean_ece; digits=4)) ± $(round(s.std_ece; digits=4))", 16),
                rpad(delta_str, 22),
                rpad(rel_str, 22),
                "$(round(s.mean_mce; digits=4))",
            )
        end
        println(io, "─" ^ 100)
        println(io)
        println(io, "ECE    = Expected Calibration Error (mean ± std over seeds)")
        println(io, "ΔECE   = paired difference: ECE_method(seed) − ECE_vanilla(seed),")
        println(io, "         reported as mean ± SE (standard error = std/√n)")
        println(io, "ΔECE%  = paired relative difference: (ECE_method − ECE_vanilla) / ECE_vanilla per seed,")
        println(io, "         reported as mean ± SE in percent")
        println(io, "MCE    = Maximum Calibration Error (mean over seeds)")
        println(io)

        # ── Paired differences per seed ──
        println(io, "─" ^ 80)
        println(io, "Paired ΔECE per seed:")
        println(io, "─" ^ 80)
        println(io, rpad("Seed", 8),
                    rpad("vanilla ECE", 14),
                    rpad("RL1 ΔECE", 14),
                    rpad("DRR ΔECE", 14),
                    "PMMP ΔECE")
        println(io, "─" ^ 80)
        for s in all_seeds
            vanilla_ece = method_seed_ece["vanilla"][s]
            rl1_delta = haskey(method_seed_ece["RL1_procedure"], s) ? method_seed_ece["RL1_procedure"][s] - vanilla_ece : NaN
            drr_delta = haskey(method_seed_ece["DRR_procedure"], s) ? method_seed_ece["DRR_procedure"][s] - vanilla_ece : NaN
            pmmp_delta = haskey(method_seed_ece["PMMP_procedure"], s) ? method_seed_ece["PMMP_procedure"][s] - vanilla_ece : NaN
            println(io,
                rpad(string(s), 8),
                rpad(string(round(vanilla_ece; digits=4)), 14),
                rpad(string(round(rl1_delta; digits=4)), 14),
                rpad(string(round(drr_delta; digits=4)), 14),
                string(round(pmmp_delta; digits=4)),
            )
        end
        println(io, "─" ^ 80)
        println(io)

        # ── Per-run detail table ──
        println(io, "─" ^ 80)
        println(io, "Per-run details:")
        println(io, "─" ^ 80)
        println(io, rpad("Method", 18), rpad("Seed", 8), rpad("Acc", 10), rpad("ECE", 10), "MCE")
        println(io, "─" ^ 80)
        sorted = sort(collect(results); by=x -> (x.second["method_name"], x.second["seed"]))
        for (name, r) in sorted
            println(io,
                rpad(r["method_name"], 18),
                rpad(string(r["seed"]), 8),
                rpad(string(round(r["accuracy"]; digits=4)), 10),
                rpad(string(round(r["ece"]; digits=4)), 10),
                string(round(r["mce"]; digits=4)),
            )
        end
        println(io, "─" ^ 80)
    end

    println("Report saved to $report_path")
end

generate_report(results, method_groups)

# ─── Print summary to stdout ─────────────────────────────────────────────────

println("\n" * "="^90)
println("ECE Benchmark Summary (CIFAR-10 Clean Test Set, $N_BINS bins)")
println("="^90)
println(rpad("Method", 20), rpad("Acc", 10), rpad("ECE", 16), rpad("ΔECE (paired)", 22), "ΔECE%")
println("-"^90)
baseline_seeds = Dict(r["seed"] => r["ece"] for r in method_groups["vanilla"])
for method in ["vanilla", "RL1_procedure", "DRR_procedure", "PMMP_procedure"]
    haskey(method_groups, method) || continue
    runs = method_groups[method]
    mean_acc = mean(r["accuracy"] for r in runs)
    mean_ece = mean(r["ece"] for r in runs)
    std_ece = length(runs) > 1 ? std([r["ece"] for r in runs]) : 0.0
    if method == "vanilla"
        println(rpad(method, 20), rpad("$(round(mean_acc; digits=4))", 10),
                rpad("$(round(mean_ece; digits=4)) ± $(round(std_ece; digits=4))", 16),
                rpad("—", 22), "—")
    else
        deltas = [r["ece"] - baseline_seeds[r["seed"]] for r in runs if haskey(baseline_seeds, r["seed"])]
        rel_deltas = [(r["ece"] - baseline_seeds[r["seed"]]) / baseline_seeds[r["seed"]] for r in runs if haskey(baseline_seeds, r["seed"])]
        n = length(deltas)
        md = mean(deltas)
        se = n > 1 ? std(deltas) / sqrt(n) : 0.0
        mr = mean(rel_deltas) * 100
        se_r = (n > 1 ? std(rel_deltas) / sqrt(n) : 0.0) * 100
        sign_str = md >= 0 ? "+" : ""
        rsign = mr >= 0 ? "+" : ""
        println(rpad(method, 20), rpad("$(round(mean_acc; digits=4))", 10),
                rpad("$(round(mean_ece; digits=4)) ± $(round(std_ece; digits=4))", 16),
                rpad("$(sign_str)$(round(md; digits=4)) ± $(round(se; digits=4))", 22),
                "$(rsign)$(round(mr; digits=1))% ± $(round(se_r; digits=1))%")
    end
end
println("="^90)
