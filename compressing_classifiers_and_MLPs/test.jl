using Pkg; Pkg.activate(".")
using Revise

###

using CUDA, JLD2, Lux, Random

# Create a simple model on GPU
model = Dense(10, 5)
ps, st = Lux.setup(Random.default_rng(), model)
ps_gpu = ps |> gpu_device()

# Try to save directly
try
    jldsave("test_gpu.jld2"; params=ps_gpu)
    println("✓ GPU tensors saved directly (unexpected!)")
catch e
    println("✗ Cannot save GPU tensors directly:")
    println("  Error: ", typeof(e))
end

# Save after moving to CPU
ps_cpu = ps_gpu |> cpu_device()
jldsave("test_cpu.jld2"; params=ps_cpu)
println("✓ CPU tensors saved successfully")

@load "test_gpu.jld2" params

###


# TODO:
# then start imagenet experiments
# continue by making sure the code executes as given in the readme.

using CompressingClassifiersMLPs: Checkpointer, TrainingArguments
using .Checkpointer
using .TrainingArguments: TrainArgs, AbstractTrainArgs

# Create a temporary directory for checkpoint files
tmpdir = mktempdir()

# Dummy training arguments and checkpoint content
args = TrainArgs{Float32}()
metadata = CheckpointMetadata(path=tmpdir)
content = CheckpointContent(args=args, tstate=nothing)

# Save an initial checkpoint (status defaults to :running)
save_checkpoint(metadata, content)

# No "available" checkpoint yet
@assert find_available_checkpoint(tmpdir) === nothing "Expected no available checkpoint"

# Mark as available and save again
metadata.status = :available
save_checkpoint(metadata, content)

# Find the available checkpoint file
filepath = find_available_checkpoint(tmpdir)
@assert filepath !== nothing "Failed to locate available checkpoint"

# Load the checkpoint – status becomes :running
md, ct = load_checkpoint(filepath)
@assert md.status == :running "Status should be :running after loading"

# Mark as finished and persist
md.status = :finished
save_checkpoint(md, ct)

# Clean finished checkpoints – file should be removed
clean_checkpoint_files(tmpdir)
@assert !isfile(filepath) "Finished checkpoint should be removed after cleaning"

println("All Checkpointer tests passed.")

