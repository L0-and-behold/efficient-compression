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
    ("vanilla",        joinpath(path_to_db, "vanilla-lr-rho-sweep_3-6", "checkpoints", "FINISHED_wild-lynx.jld2")),
    ("DRR_procedure",  joinpath(path_to_db, "alpha-sweep-v1_9-13", "checkpoints", "FINISHED_happy-ibis.jld2")),
    ("RL1_procedure",  joinpath(path_to_db, "further-RL1-points_2-2", "checkpoints", "FINISHED_wry-osprey.jld2")),
    ("PMMP_procedure", joinpath(path_to_db, "alpha-pmmp-v1_9-10", "checkpoints", "FINISHED_fierce-fox.jld2")),
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
    println("\n─── $method_name ───")
    println("  Loading checkpoint: $ckpt_path")
    @assert isfile(ckpt_path) "Checkpoint not found: $ckpt_path"

    metadata, content = load_checkpoint(ckpt_path, args)
    tstate = content.tstate

    println("  Running forward pass over validation set...")
    logits, labels = collect_logits_and_labels(tstate, val_set)
    rd = reliability_diagram_data(logits, labels; n_bins=N_BINS)

    results[method_name] = Dict(
        "method_name" => method_name,
        "checkpoint" => basename(ckpt_path),
        "ece" => rd.ece,
        "mce" => rd.mce,
        "bin_accuracies" => replace(rd.bin_accuracies, NaN32 => -1f0),
        "bin_confidences" => replace(rd.bin_confidences, NaN32 => -1f0),
        "bin_counts" => rd.bin_counts,
    )

    println("  ECE = $(round(rd.ece; digits=4)), MCE = $(round(rd.mce; digits=4))")
end

# ─── Save numerical results ──────────────────────────────────────────────────

open(joinpath(output_dir, "ece_imagenet_results.json"), "w") do io
    JSON3.pretty(io, results)
end
println("\nResults saved to $(joinpath(output_dir, "ece_imagenet_results.json"))")

# ─── Generate report ─────────────────────────────────────────────────────────

report_path = joinpath(output_dir, "ece_imagenet_report.out")
open(report_path, "w") do io
    println(io, "=" ^ 90)
    println(io, "  ECE Benchmark Report — ImageNet Validation Set ($N_BINS bins)")
    println(io, "  Generated: $(Dates.now())")
    println(io, "=" ^ 90)
    println(io)

    println(io, "─" ^ 90)
    println(io, rpad("Method", 18), rpad("Checkpoint", 24), rpad("ECE", 12), "MCE")
    println(io, "─" ^ 90)

    vanilla_ece = results["vanilla"]["ece"]
    for method in ["vanilla", "DRR_procedure", "RL1_procedure", "PMMP_procedure"]
        haskey(results, method) || continue
        r = results[method]
        delta_str = method == "vanilla" ? "—" : begin
            d = r["ece"] - vanilla_ece
            sign_str = d >= 0 ? "+" : ""
            "$(sign_str)$(round(d; digits=4))"
        end
        println(io,
            rpad(method, 18),
            rpad(r["checkpoint"], 24),
            rpad("$(round(r["ece"]; digits=4))", 12),
            "$(round(r["mce"]; digits=4))",
        )
    end
    println(io, "─" ^ 90)
    println(io)

    # ΔECE relative to vanilla
    println(io, "─" ^ 90)
    println(io, rpad("Method", 18), rpad("ΔECE (vs vanilla)", 22), "ΔECE%")
    println(io, "─" ^ 90)
    for method in ["vanilla", "DRR_procedure", "RL1_procedure", "PMMP_procedure"]
        haskey(results, method) || continue
        if method == "vanilla"
            println(io, rpad(method, 18), rpad("— (baseline)", 22), "—")
        else
            d = results[method]["ece"] - vanilla_ece
            rel = d / vanilla_ece * 100
            sign_str = d >= 0 ? "+" : ""
            rsign = rel >= 0 ? "+" : ""
            println(io, rpad(method, 18), rpad("$(sign_str)$(round(d; digits=4))", 22), "$(rsign)$(round(rel; digits=1))%")
        end
    end
    println(io, "─" ^ 90)
end
println("Report saved to $report_path")

# ─── Print summary to stdout ─────────────────────────────────────────────────

println("\n" * "="^70)
println("ECE Benchmark Summary (ImageNet Val, $N_BINS bins)")
println("="^70)
println(rpad("Method", 18), rpad("ECE", 12), rpad("MCE", 12), rpad("ΔECE", 12), "ΔECE%")
println("-"^70)
vanilla_ece = results["vanilla"]["ece"]
for method in ["vanilla", "DRR_procedure", "RL1_procedure", "PMMP_procedure"]
    haskey(results, method) || continue
    r = results[method]
    if method == "vanilla"
        println(rpad(method, 18), rpad("$(round(r["ece"]; digits=4))", 12), rpad("$(round(r["mce"]; digits=4))", 12), rpad("—", 12), "—")
    else
        d = r["ece"] - vanilla_ece
        rel = d / vanilla_ece * 100
        sign_str = d >= 0 ? "+" : ""
        rsign = rel >= 0 ? "+" : ""
        println(rpad(method, 18), rpad("$(round(r["ece"]; digits=4))", 12), rpad("$(round(r["mce"]; digits=4))", 12), rpad("$(sign_str)$(round(d; digits=4))", 12), "$(rsign)$(round(rel; digits=1))%")
    end
end
println("="^70)

# ─── Reliability diagram PNGs ────────────────────────────────────────────────

for method in ["vanilla", "DRR_procedure", "RL1_procedure", "PMMP_procedure"]
    haskey(results, method) || continue
    r = results[method]

    bin_edges = range(0.0, 1.0, length=N_BINS + 1)
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in 1:N_BINS]
    bin_width = 1.0 / N_BINS

    acc = [a >= 0 ? a : NaN for a in r["bin_accuracies"]]
    conf = [c >= 0 ? c : NaN for c in r["bin_confidences"]]

    p = Plots.plot(
        size=(600, 500),
        title="Reliability Diagram: $method (ImageNet)\n(ECE = $(round(r["ece"]; digits=4)))",
        xlabel="Confidence", ylabel="Accuracy",
        xlim=(0, 1), ylim=(0, 1),
        legend=:topleft, grid=true, framestyle=:box,
    )
    Plots.plot!(p, [0, 1], [0, 1]; linestyle=:dash, color=:gray, label="Perfect calibration")
    Plots.bar!(p, bin_centers, acc; width=bin_width * 0.8, alpha=0.7, color=:steelblue, label="Model")
    Plots.savefig(p, joinpath(output_dir, "reliability_$(method).png"))
    println("Saved reliability diagram: reliability_$(method).png")
end
