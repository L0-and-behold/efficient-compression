"""
    generate_dummy_models.jl

Creates dummy VGG model BSON files for testing the ECE benchmark pipeline.
Generates 4 fake "runs" — one per method (vanilla, RL1, DRR, PMMP) —
in the same folder structure that run_ECE_benchmark.jl expects.
"""

using Pkg; Pkg.activate("."); using Revise

begin
    using Lux
    using Random
    using CSV, DataFrames
    using BSON
end
using CompressingClassifiersMLPs

using CompressingClassifiersMLPs.TrainingTools: save_train_state, to_cpu
using CompressingClassifiersMLPs.OptimizationProcedures: VGG
using Optimisers

# ─── Config ──────────────────────────────────────────────────────────────────

output_base = "./src/DatasetsModels/CIFAR-C/tested_models/sweep/1/"
mkpath(output_base)

# Define dummy runs: (folder_suffix, method, α, β, initial_p_value, seed)
dummy_runs = [
    ("run_1-001", "RL1_procedure",  0f0, 0f0, 0f0, 1234),   # vanilla
    ("run_2-002", "RL1_procedure",  1f-3, 1f0, 0f0, 1234),   # RL1
    ("run_3-003", "DRR_procedure",  1f-3, 1f0, 0f0, 1234),   # DRR
    ("run_4-004", "PMMP_procedure", 1f-3, 1f0, 0.5f0, 1234), # PMMP
]

for (folder_name, method, α, β, p_val, seed) in dummy_runs
    println("Creating dummy model: $folder_name ($method)")

    # Create folder structure
    run_dir = joinpath(output_base, folder_name)
    artifacts_dir = joinpath(run_dir, "artifacts", "run-dummy")
    mkpath(artifacts_dir)

    # Generate a VGG model with random weights (untrained)
    model = VGG()
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    ps, st = Lux.setup(rng, model)  # CPU only for dummy
    tstate = Lux.Training.TrainState(model, ps, st, Optimisers.Adam(1f-3))

    # Save as BSON (same format as real trained models)
    save_train_state(tstate, model, rng, joinpath(artifacts_dir, "train_state.bson"))

    # Create a CSV with the metadata that run_ECE_benchmark.jl reads
    df = DataFrame(
        optimization_procedure = [method],
        α = [α],
        β = [β],
        initial_p_value = [p_val],
        seed = [seed],
    )
    CSV.write(joinpath(run_dir, "run_info.csv"), df)

    println("  Saved to $run_dir")
end

println("\nDone. Dummy models written to $output_base")
