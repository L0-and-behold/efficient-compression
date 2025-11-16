using Pkg
Pkg.activate(".")
Pkg.instantiate()

include("src/DatasetsModels/DatasetsModels.jl")
include("src/OptimizationProcedures/OptimizationProcedures.jl")
include("src/TrainArgs.jl")

using CUDA
using Lux
using Optimisers
using LuxCUDA
using Statistics
using TOML
using Suppressor
using Random
using Revise
using .DatasetsModels: imagenet_data, toy_imagenet_data
using .OptimizationProcedures: lux_training!, resnet

# ------------------------------------------------------------
# Utility to print which GPU device is being used
function print_gpu_info(msg::String)
    device = CUDA.device()
    println("[GPU INFO] ", msg, " - device: ", device, " (", CUDA.name(device), ")")
end

# Load configuration (same as in original script)
cfg = TOML.parsefile("config.toml")
imagenet_path = cfg["paths"]["imagenet_path"]
println("ImageNet path: ", imagenet_path)

# -----------------------------------------------------------------
# Create model and training state
println("Creating model (ResNet‑50)...")
model = resnet(; depth=18)
# optimiser state
# initialise parameters on GPU
opt_state = Momentum(0.1f0, 0.9f0)
rng = Random.default_rng()
Random.seed!(rng, 1234)
ps, st = Lux.setup(rng, model) |> Lux.gpu_device()
tstate = Lux.Training.TrainState(model, ps, st, opt_state)
# simple loss – cross‑entropy for ImageNet (1000 classes)
loss_fun = Lux.CrossEntropyLoss(; logits=Val(true))

# -----------------------------------------------------------------
# Load data (small subset for debugging)
println("Loading a small subset of (toy-)ImageNet data…")
imagenet_data_function = trainbatchsize -> toy_imagenet_data(imagenet_path, trainbatchsize, trainbatchsize, 224; dev=gpu_device())
train_set, val_set, test_set = imagenet_data_function(32)
println("# training batches: ", length(train_set))
print_gpu_info("After data loading")

# -----------------------------------------------------------------
# Run a minimal training loop
println("Starting training…")
vjp = AutoZygote()
for (i, batch) in enumerate(train_set)
    global tstate  
    batch_time = time()
    println("▶ batch $i  start update-step")
    grads, loss, _, tstate = Training.compute_gradients(vjp, loss_fun, batch, tstate)
    tstate = Training.apply_gradients!(tstate, grads)
    println("▶ batch $i  update-step finished after $(time() - batch_time) s. Loss: $loss")
end

println("Training finished.")
println("Final training loss: ", logs["train_loss"][end])
println("Final validation loss: ", logs["val_loss"][end])
print_gpu_info("End of script")

# End of file
