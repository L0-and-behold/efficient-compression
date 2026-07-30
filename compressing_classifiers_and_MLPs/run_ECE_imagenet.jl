"""
    run_ECE_imagenet.jl

Expected Calibration Error (ECE) benchmark for ImageNet models.
Evaluates calibration of baseline (vanilla) and compressed models (RL1, DRR, PMMP)
on the ImageNet validation set (50k images). Produces:
  - JSON file with numerical ECE/MCE results
  - .out report with summary table
  - Reliability diagram PNGs

Uses the 4 paper-reported checkpoints (FINISHED .jld2 files).
"""

using Pkg; Pkg.activate("."); using Revise

begin
    using Lux
    using CUDA
    using Random
    using Statistics: mean, std
    using JSON3
    using JLD2
    using Dates
    using Plots
end
using CompressingClassifiersMLPs

using CompressingClassifiersMLPs.Config: load_imagenet_config
using CompressingClassifiersMLPs.TrainingArguments: TrainArgs, AbstractTrainArgs
using CompressingClassifiersMLPs.OptimizationProcedures: PMMP_procedure, RL1_procedure, DRR_procedure, resnet50, testmode_states, collect_logits_and_labels, reliability_diagram_data
using CompressingClassifiersMLPs.DatasetsModels: imagenet_data_function
using CompressingClassifiersMLPs.Checkpointer: load_checkpoint

using JLD2

# Handle old checkpoints missing the scale_alpha_with_lr field in TrainArgs
function JLD2.rconvert(::Type{<:AbstractTrainArgs}, x)
    # x is a ReconstructedMutable with the old fields; build a fresh TrainArgs
    # and copy over whichever fields exist
    args = TrainArgs{Float32}()
    for field in fieldnames(typeof(x))
        if hasfield(typeof(args), field)
            try
                setfield!(args, field, getfield(x, field))
            catch
            end
        end
    end
    return args
end

# ─── Configuration ───────────────────────────────────────────────────────────

N_BINS = 15
output_dir = "./experiment-results/ece_imagenet/"
mkpath(output_dir)

# Paper-reported checkpoints: (method_name, checkpoint_path_relative_to_experiment_data)
path_to_db, _, _ = load_imagenet_config()
CHECKPOINTS = [
    # Vanilla baseline
    ("vanilla",        joinpath(path_to_db, "vanilla-lr-rho-sweep_3-6", "checkpoints", "FINISHED_wild-lynx.jld2")),
    # DRR (2 checkpoints)
    ("DRR_procedure",  joinpath(path_to_db, "alpha-sweep-v1_9-13", "checkpoints", "FINISHED_happy-ibis.jld2")),
    ("DRR_procedure",  joinpath(path_to_db, "alpha-sweep-v1_7-13", "checkpoints", "FINISHED_fierce-otter.jld2")),
    # PMMP (7 checkpoints)
    ("PMMP_procedure", joinpath(path_to_db, "pmmp-alpha-rho-v1_9-9", "checkpoints", "FINISHED_quiet-koala.jld2")),
    ("PMMP_procedure", joinpath(path_to_db, "alpha-pmmp-v1_5-6", "checkpoints", "FINISHED_lunar-deer.jld2")),
    ("PMMP_procedure", joinpath(path_to_db, "alpha-pmmp-v1_12-12", "checkpoints", "FINISHED_gentle-quokka.jld2")),
    ("PMMP_procedure", joinpath(path_to_db, "alpha-pmmp-v1_9-10", "checkpoints", "FINISHED_fierce-fox.jld2")),
    ("PMMP_procedure", joinpath(path_to_db, "pmmp-alpha-rho-v1_10-13", "checkpoints", "FINISHED_umber-dingo.jld2")),
    ("PMMP_procedure", joinpath(path_to_db, "pmmp-alpha-rho-v1_11-13", "checkpoints", "FINISHED_keen-impala.jld2")),
    ("PMMP_procedure", joinpath(path_to_db, "alpha-pmmp-v1_10-10", "checkpoints", "FINISHED_gentle-quail.jld2")),
    # RL1 (6 checkpoints)
    ("RL1_procedure",  joinpath(path_to_db, "further-RL1-points_2-2", "checkpoints", "FINISHED_wry-osprey.jld2")),
    ("RL1_procedure",  joinpath(path_to_db, "further-RL1-points_1-2", "checkpoints", "FINISHED_gentle-hawk.jld2")),
    ("RL1_procedure",  joinpath(path_to_db, "alpha-sweep-v1_4-13", "checkpoints", "FINISHED_zesty-quail.jld2")),
    ("RL1_procedure",  joinpath(path_to_db, "alpha-sweep-v1_2-13", "checkpoints", "FINISHED_quiet-deer.jld2")),
    ("RL1_procedure",  joinpath(path_to_db, "alpha-sweep-v1_14-14", "checkpoints", "FINISHED_proud-osprey.jld2")),
    ("RL1_procedure",  joinpath(path_to_db, "alpha-sweep-v1_3-13", "checkpoints", "FINISHED_ruddy-kestrel.jld2")),
]

# ─── Set up args (needed for checkpoint loading — moves tstate to device) ────

args = TrainArgs{Float32}()
args.architecture = resnet50
args.dataset = imagenet_data_function()
args.dev = Lux.gpu_device()
args.val_batch_size = 128

# ─── Load ImageNet validation set ────────────────────────────────────────────

println("Loading ImageNet validation set...")
_, val_set, _ = args.dataset(args.val_batch_size)
println("  Done. Val set loaded.")

# ─── Evaluate each checkpoint ────────────────────────────────────────────────

results = Dict{String, Any}()

for (method_name, ckpt_path) in CHECKPOINTS
    animal = match(r"FINISHED_(.+)\.jld2", basename(ckpt_path)).captures[1]
    key = "$(method_name)__$(animal)"
    println("\n─── $method_name ($animal) ───")
    println("  Loading checkpoint: $ckpt_path")
    @assert isfile(ckpt_path) "Checkpoint not found: $ckpt_path"

    metadata, content = load_checkpoint(ckpt_path, args)
    tstate = content.tstate

    println("  Running forward pass over validation set...")
    logits, labels = collect_logits_and_labels(tstate, val_set)
    rd = reliability_diagram_data(logits, labels; n_bins=N_BINS)

    results[key] = Dict(
        "method_name" => method_name,
        "animal" => animal,
        "checkpoint" => basename(ckpt_path),
        "ece" => rd.ece,
        "mce" => rd.mce,
        "bin_accuracies" => replace(rd.bin_accuracies, NaN32 => -1f0),
        "bin_confidences" => replace(rd.bin_confidences, NaN32 => -1f0),
        "bin_counts" => rd.bin_counts,
    )

    println("  ECE = $(round(rd.ece; digits=4)), MCE = $(round(rd.mce; digits=4))")

    # Free GPU memory
    GC.gc(true)
    CUDA.reclaim()
end

# ─── Save numerical results ──────────────────────────────────────────────────

open(joinpath(output_dir, "ece_imagenet_results.json"), "w") do io
    JSON3.pretty(io, results)
end
println("\nResults saved to $(joinpath(output_dir, "ece_imagenet_results.json"))")

# ─── Group by method and compute statistics ──────────────────────────────────

method_groups = Dict{String, Vector{Dict}}()
for (_, r) in results
    method = r["method_name"]
    push!(get!(method_groups, method, Dict[]), r)
end

vanilla_ece = method_groups["vanilla"][1]["ece"]

# ─── Generate report ─────────────────────────────────────────────────────────

report_path = joinpath(output_dir, "ece_imagenet_report.out")
open(report_path, "w") do io
    println(io, "=" ^ 100)
    println(io, "  ECE Benchmark Report — ImageNet Validation Set ($N_BINS bins)")
    println(io, "  Generated: $(Dates.now())")
    println(io, "  Total checkpoints evaluated: $(length(results))")
    println(io, "=" ^ 100)
    println(io)

    # Per-run detail table
    println(io, "─" ^ 100)
    println(io, rpad("Method", 18), rpad("Animal", 18), rpad("ECE", 12), rpad("MCE", 12), "ΔECE vs vanilla")
    println(io, "─" ^ 100)
    for method in ["vanilla", "DRR_procedure", "RL1_procedure", "PMMP_procedure"]
        haskey(method_groups, method) || continue
        for r in sort(method_groups[method]; by=x -> x["ece"])
            d = r["ece"] - vanilla_ece
            delta_str = method == "vanilla" ? "—" : begin
                sign_str = d >= 0 ? "+" : ""
                "$(sign_str)$(round(d; digits=4))"
            end
            println(io, rpad(method, 18), rpad(r["animal"], 18),
                    rpad("$(round(r["ece"]; digits=4))", 12),
                    rpad("$(round(r["mce"]; digits=4))", 12), delta_str)
        end
    end
    println(io, "─" ^ 100)
    println(io)

    # Aggregate summary: mean ± std per method
    println(io, "─" ^ 100)
    println(io, rpad("Method", 18), rpad("n", 4), rpad("ECE (mean±std)", 22), rpad("ΔECE (mean±SE)", 22), "ΔECE%")
    println(io, "─" ^ 100)
    for method in ["vanilla", "DRR_procedure", "RL1_procedure", "PMMP_procedure"]
        haskey(method_groups, method) || continue
        eces = [r["ece"] for r in method_groups[method]]
        n = length(eces)
        mean_ece = mean(eces)
        std_ece = n > 1 ? std(eces) : 0.0
        if method == "vanilla"
            println(io, rpad(method, 18), rpad(string(n), 4),
                    rpad("$(round(mean_ece; digits=4))", 22),
                    rpad("— (baseline)", 22), "—")
        else
            deltas = eces .- vanilla_ece
            mean_d = mean(deltas)
            se_d = n > 1 ? std(deltas) / sqrt(n) : 0.0
            rel = mean_d / vanilla_ece * 100
            sign_str = mean_d >= 0 ? "+" : ""
            rsign = rel >= 0 ? "+" : ""
            println(io, rpad(method, 18), rpad(string(n), 4),
                    rpad("$(round(mean_ece; digits=4)) ± $(round(std_ece; digits=4))", 22),
                    rpad("$(sign_str)$(round(mean_d; digits=4)) ± $(round(se_d; digits=4))", 22),
                    "$(rsign)$(round(rel; digits=1))%")
        end
    end
    println(io, "─" ^ 100)
end
println("Report saved to $report_path")

# ─── Print summary to stdout ─────────────────────────────────────────────────

println("\n" * "="^90)
println("ECE Benchmark Summary (ImageNet Val, $N_BINS bins, $(length(results)) checkpoints)")
println("="^90)
println(rpad("Method", 18), rpad("n", 4), rpad("ECE (mean±std)", 22), rpad("ΔECE (mean±SE)", 22), "ΔECE%")
println("-"^90)
for method in ["vanilla", "DRR_procedure", "RL1_procedure", "PMMP_procedure"]
    haskey(method_groups, method) || continue
    eces = [r["ece"] for r in method_groups[method]]
    n = length(eces)
    mean_ece = mean(eces)
    std_ece = n > 1 ? std(eces) : 0.0
    if method == "vanilla"
        println(rpad(method, 18), rpad(string(n), 4), rpad("$(round(mean_ece; digits=4))", 22), rpad("—", 22), "—")
    else
        deltas = eces .- vanilla_ece
        mean_d = mean(deltas)
        se_d = n > 1 ? std(deltas) / sqrt(n) : 0.0
        rel = mean_d / vanilla_ece * 100
        sign_str = mean_d >= 0 ? "+" : ""
        rsign = rel >= 0 ? "+" : ""
        println(rpad(method, 18), rpad(string(n), 4),
                rpad("$(round(mean_ece; digits=4)) ± $(round(std_ece; digits=4))", 22),
                rpad("$(sign_str)$(round(mean_d; digits=4)) ± $(round(se_d; digits=4))", 22),
                "$(rsign)$(round(rel; digits=1))%")
    end
end
println("="^90)

# ─── Reliability diagram PNGs (averaged per method) ──────────────────────────

for method in ["vanilla", "DRR_procedure", "RL1_procedure", "PMMP_procedure"]
    haskey(method_groups, method) || continue
    runs = method_groups[method]

    bin_edges = range(0.0, 1.0, length=N_BINS + 1)
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in 1:N_BINS]
    bin_width = 1.0 / N_BINS

    # Average bin accuracies over all runs (ignoring empty bins marked as -1)
    avg_acc = fill(NaN, N_BINS)
    for b in 1:N_BINS
        valid = [r["bin_accuracies"][b] for r in runs if r["bin_accuracies"][b] >= 0]
        if !isempty(valid)
            avg_acc[b] = mean(valid)
        end
    end

    mean_ece = mean(r["ece"] for r in runs)

    p = Plots.plot(
        size=(600, 500),
        title="Reliability Diagram: $method (ImageNet, n=$(length(runs)))\n(mean ECE = $(round(mean_ece; digits=4)))",
        xlabel="Confidence", ylabel="Accuracy",
        xlim=(0, 1), ylim=(0, 1),
        legend=:topleft, grid=true, framestyle=:box,
    )
    Plots.plot!(p, [0, 1], [0, 1]; linestyle=:dash, color=:gray, label="Perfect calibration")
    Plots.bar!(p, bin_centers, avg_acc; width=bin_width * 0.8, alpha=0.7, color=:steelblue, label="Model (avg)")
    Plots.savefig(p, joinpath(output_dir, "reliability_$(method).png"))
    println("Saved reliability diagram: reliability_$(method).png")
end
