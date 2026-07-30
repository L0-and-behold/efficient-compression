"""
    run_ECE_benchmark.jl

Expected Calibration Error (ECE) benchmark for compressed neural networks.
Evaluates calibration of baseline (vanilla) and compressed models (RL1, DRR, PMMP).

Supports both CIFAR-10 and ImageNet by changing the USER CONFIGURATION block below.
Models are loaded from a directory of experiment subfolders, each containing a
runs.csv and either BSON artifacts or JLD2 checkpoints.

Produces:
  - JSON file with numerical ECE/MCE results
  - .out report with summary table
  - Reliability diagram PNGs

Reference: Guo et al. 2017, "On Calibration of Modern Neural Networks"
"""

using Pkg; Pkg.activate("."); using Revise

begin
    using Lux
    using CUDA
    using Random
    using Statistics: mean, std
    using JSON3
    using CSV, DataFrames
    using BSON
    using JLD2
    using Optimisers
    using Dates
    using Plots
end
using CompressingClassifiersMLPs

using CompressingClassifiersMLPs.TrainingArguments: TrainArgs, AbstractTrainArgs
using CompressingClassifiersMLPs.TrainingTools: load_train_state, to_gpu
using CompressingClassifiersMLPs.OptimizationProcedures: PMMP_procedure, RL1_procedure, DRR_procedure, Lenet_MLP, Lenet_5, Lenet_5_Caffe, VGG, resnet50, logitcrossentropy, logitcrossentropy_ls, generate_tstate, accuracy, initialize_DRR_loss, initialize_RL1_loss, initialize_PMMP_loss, testmode_states, collect_logits_and_labels, expected_calibration_error, reliability_diagram_data
using CompressingClassifiersMLPs.DatasetsModels: MNIST_data, CIFAR_data, imagenet_data_function
using CompressingClassifiersMLPs.BatchRun: do_batch_run, get_sub_batch, single_run_routine_classifier
using CompressingClassifiersMLPs.Checkpointer: load_checkpoint
using CompressingClassifiersMLPs.Config: load_imagenet_config, load_ece_config

# Handle old JLD2 checkpoints where TrainArgs has missing fields (e.g. scale_alpha_with_lr)
function JLD2.rconvert(::Type{<:AbstractTrainArgs}, x)
    args = TrainArgs{Float32}()
    for field in fieldnames(typeof(x))
        if hasfield(typeof(args), field)
            try setfield!(args, field, getfield(x, field)) catch end
        end
    end
    return args
end

# ══════════════════════════════════════════════════════════════════════════════
# ─── USER CONFIGURATION ──────────────────────────────────────────────────────
# Edit this block to switch between CIFAR-10 and ImageNet.
# ══════════════════════════════════════════════════════════════════════════════

# ── Option A: CIFAR-10 (directory scan mode) ──
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
DATASET_NAME = "CIFAR-10"
model_path = load_ece_config()  # reads ece_cifar_model_path from config.toml
output_dir = "./experiment-results/ece_cifar/"
eval_set_fn() = begin _, _, test = CIFAR_data(args.train_batch_size, args.dev; seed=1234); test end
CHECKPOINTS = nothing  # use directory scan

# ── Option B: ImageNet (explicit checkpoint list) ──
# begin
#     args = TrainArgs{Float32}()
#     args.architecture = resnet50
#     args.dataset = imagenet_data_function()
#     args.dev = Lux.gpu_device()
#     args.val_batch_size = 128
# end
# DATASET_NAME = "ImageNet"
# output_dir = "./experiment-results/ece_imagenet/"
# eval_set_fn() = begin _, val, _ = args.dataset(args.val_batch_size); val end
# model_path = nothing  # not used in checkpoint-list mode
#
# # Explicit checkpoint list: (method_name, path_to_jld2)
# path_to_db, _, _ = load_imagenet_config()
# CHECKPOINTS = [
#     ("vanilla",        ""),
#     ("DRR_procedure",  ""),
#     ("RL1_procedure",  ""),
#     ("PMMP_procedure", ""),
# ]

# ── Shared settings ──
N_BINS = 15
MIN_ACCURACY = 0.0   # set > 0 to exclude low-accuracy runs from aggregate stats

# ══════════════════════════════════════════════════════════════════════════════
# ─── END USER CONFIGURATION ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

mkpath(output_dir)

# ─── Load evaluation set ─────────────────────────────────────────────────────

println("Loading $DATASET_NAME evaluation set...")
eval_set = eval_set_fn()
println("  Done.")

# ─── Helper: load model from a subfolder (BSON or JLD2) ─────────────────────

function load_model_from_subfolder(subfolder::String, args)
    # Try BSON first: artifacts/<run-id>/train_state.bson
    artifacts_dir = joinpath(subfolder, "artifacts")
    if isdir(artifacts_dir)
        run_dirs = readdir(artifacts_dir; join=true)
        if !isempty(run_dirs)
            bson_path = joinpath(run_dirs[1], "train_state.bson")
            if isfile(bson_path)
                tstate_cpu, _, _ = load_train_state(bson_path)
                return to_gpu(tstate_cpu)
            end
        end
    end

    # Fallback: checkpoints/FINISHED_*.jld2
    ckpt_dir = joinpath(subfolder, "checkpoints")
    if isdir(ckpt_dir)
        jld2_files = filter(f -> startswith(f, "FINISHED_") && endswith(f, ".jld2"), readdir(ckpt_dir))
        if !isempty(jld2_files)
            ckpt_path = joinpath(ckpt_dir, jld2_files[1])
            _, content = load_checkpoint(ckpt_path, args)
            return content.tstate
        end
    end

    return nothing  # no model found
end

# ─── Evaluate models ─────────────────────────────────────────────────────────

results = Dict{String, Any}()

if !isnothing(CHECKPOINTS)
    # ── Checkpoint-list mode (ImageNet) ──
    for (method_name, ckpt_path) in CHECKPOINTS
        animal = match(r"FINISHED_(.+)\.jld2", basename(ckpt_path)).captures[1]
        storage_name = "$(method_name)__$(animal)"
        println("Loading: $storage_name")
        @assert isfile(ckpt_path) "Checkpoint not found: $ckpt_path"

        _, content = load_checkpoint(ckpt_path, args)
        tstate = content.tstate

        logits, labels = collect_logits_and_labels(tstate, eval_set)
        rd = reliability_diagram_data(logits, labels; n_bins=N_BINS)

        results[storage_name] = Dict(
            "method_name" => method_name,
            "seed" => 0,  # single checkpoint per method
            "alpha" => 0f0,
            "ece" => rd.ece,
            "mce" => rd.mce,
            "accuracy" => 0f0,  # skip full accuracy pass (ECE is the focus)
            "bin_accuracies" => replace(rd.bin_accuracies, NaN32 => -1f0),
            "bin_confidences" => replace(rd.bin_confidences, NaN32 => -1f0),
            "bin_counts" => rd.bin_counts,
        )

        println("  ECE = $(round(rd.ece; digits=4)), MCE = $(round(rd.mce; digits=4))")
        println()

        GC.gc(true)
        CUDA.reclaim()
    end
else
    # ── Directory-scan mode (CIFAR) ──
    for entry in readdir(model_path)
        subfolder = joinpath(model_path, entry)
        isdir(subfolder) || continue

        m = match(r"_(\d+)-\d+", entry)
        isnothing(m) && continue
        subfolder_index = m.captures[1]

        csv_files = filter(f -> endswith(f, ".csv"), readdir(subfolder))
        isempty(csv_files) && continue
        df = CSV.read(joinpath(subfolder, csv_files[1]), DataFrame)
        row = df[1, :]

        if row.optimization_procedure == "RL1_procedure" && row.α == 0f0 && row.β == 0f0 && row.initial_p_value == 0f0
            method_name = "vanilla"
        else
            method_name = row.optimization_procedure
        end
        storage_name = "$(method_name)_seed-$(row.seed)_alpha-$(row.α)_index-$(subfolder_index)"

        println("Loading: $storage_name")

        tstate = load_model_from_subfolder(subfolder, args)
        if isnothing(tstate)
            println("  ⚠ No model found in $entry, skipping.")
            continue
        end

        logits, labels = collect_logits_and_labels(tstate, eval_set)
        rd = reliability_diagram_data(logits, labels; n_bins=N_BINS)

        results[storage_name] = Dict(
            "method_name" => method_name,
            "seed" => row.seed,
            "alpha" => row.α,
            "ece" => rd.ece,
            "mce" => rd.mce,
            "accuracy" => accuracy(tstate, eval_set),
            "bin_accuracies" => replace(rd.bin_accuracies, NaN32 => -1f0),
            "bin_confidences" => replace(rd.bin_confidences, NaN32 => -1f0),
            "bin_counts" => rd.bin_counts,
        )

        println("  ECE = $(round(rd.ece; digits=4)), MCE = $(round(rd.mce; digits=4))")
        println()

        GC.gc(true)
        CUDA.reclaim()
    end
end

# ─── Save numerical results ──────────────────────────────────────────────────

open(joinpath(output_dir, "ece_results.json"), "w") do io
    JSON3.pretty(io, results)
end
println("Results saved to $(joinpath(output_dir, "ece_results.json"))")

# ─── Group results by method (dropping seeds where any method failed) ─────────

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
    push!(get!(method_groups, method, Dict[]), r)
end

if isempty(method_groups) || !haskey(method_groups, "vanilla")
    println("ERROR: No runs remaining after filtering (MIN_ACCURACY=$MIN_ACCURACY). Lower the threshold or add more data.")
    exit(1)
end

# ─── Generate .out report ────────────────────────────────────────────────────

function generate_report(results, method_groups; output_dir=output_dir, n_bins=N_BINS, dataset_name=DATASET_NAME)
    report_path = joinpath(output_dir, "ece_report.out")

    # Build seed → ECE lookup per method
    method_seed_ece = Dict{String, Dict{Any, Float64}}()
    method_seed_acc = Dict{String, Dict{Any, Float64}}()
    for (method, runs) in method_groups
        method_seed_ece[method] = Dict(r["seed"] => r["ece"] for r in runs)
        method_seed_acc[method] = Dict(r["seed"] => r["accuracy"] for r in runs)
    end

    all_seeds = sort(collect(keys(method_seed_ece["vanilla"])))

    # Paired differences: δ(s) = ECE_method(s) - ECE_vanilla(s)
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
        println(io, "  ECE Benchmark Report — $dataset_name ($n_bins bins)")
        println(io, "  Generated: $(Dates.now())")
        println(io, "  Seeds: $(all_seeds)")
        println(io, "  Excluded seeds (any method acc < $(MIN_ACCURACY)): $(isempty(excluded_seeds) ? "none" : excluded_seeds)")
        println(io, "=" ^ 100)
        println(io)

        # Main summary table
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

        # Paired differences per seed
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
            rl1_delta = haskey(method_seed_ece, "RL1_procedure") && haskey(method_seed_ece["RL1_procedure"], s) ? method_seed_ece["RL1_procedure"][s] - vanilla_ece : NaN
            drr_delta = haskey(method_seed_ece, "DRR_procedure") && haskey(method_seed_ece["DRR_procedure"], s) ? method_seed_ece["DRR_procedure"][s] - vanilla_ece : NaN
            pmmp_delta = haskey(method_seed_ece, "PMMP_procedure") && haskey(method_seed_ece["PMMP_procedure"], s) ? method_seed_ece["PMMP_procedure"][s] - vanilla_ece : NaN
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

        # Per-run detail table
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
println("ECE Benchmark Summary ($DATASET_NAME, $N_BINS bins)")
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

# ─── Reliability diagram PNGs ────────────────────────────────────────────────

function plot_reliability_diagram(method_name, method_results; output_dir=output_dir, n_bins=N_BINS)
    bin_edges = range(0.0, 1.0, length=n_bins + 1)
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in 1:n_bins]
    bin_width = 1.0 / n_bins

    all_accuracies = [r["bin_accuracies"] for r in method_results]
    all_confidences = [r["bin_confidences"] for r in method_results]
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
    n_runs = length(eces)
    title_suffix = n_runs > 1 ? " (n=$n_runs)" : ""

    p = Plots.plot(
        size=(600, 500),
        title="Reliability Diagram: $method_name$title_suffix\n(ECE = $(round(mean_ece; digits=4)))",
        xlabel="Confidence",
        ylabel="Accuracy",
        xlim=(0, 1), ylim=(0, 1),
        legend=:topleft, grid=true, framestyle=:box,
    )

    Plots.plot!(p, [0, 1], [0, 1], linestyle=:dash, color=:gray, linewidth=1.5, label="Perfect calibration")

    non_nan = .!isnan.(avg_acc)
    Plots.bar!(p, bin_centers[non_nan], avg_acc[non_nan],
        bar_width=bin_width * 0.9, color=:steelblue, alpha=0.7, label="Outputs")

    # Gap shading
    for b in 1:n_bins
        if !isnan(avg_acc[b]) && !isnan(avg_conf[b])
            gap_color = avg_conf[b] > avg_acc[b] ? :salmon : :lightgreen
            bar_lo = min(avg_acc[b], avg_conf[b])
            bar_hi = max(avg_acc[b], avg_conf[b])
            if bar_hi - bar_lo > 1e-4
                Plots.plot!(p,
                    Plots.Shape([
                        bin_centers[b] - bin_width*0.45, bin_centers[b] + bin_width*0.45,
                        bin_centers[b] + bin_width*0.45, bin_centers[b] - bin_width*0.45
                    ], [bar_lo, bar_lo, bar_hi, bar_hi]),
                    fillcolor=gap_color, fillalpha=0.5, linecolor=:transparent,
                    label=(b == findfirst(!isnan, avg_acc) ? "Gap" : ""),
                )
            end
        end
    end

    Plots.savefig(p, joinpath(output_dir, "reliability_diagram_$(method_name).png"))
    println("Saved: reliability_diagram_$(method_name).png")
end

for (method_name, method_results) in method_groups
    plot_reliability_diagram(method_name, method_results)
end
