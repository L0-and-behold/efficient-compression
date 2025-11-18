using Pkg
Pkg.activate(".")

# Try to load MPI, fall back to single-GPU mode
const USE_DISTRIBUTED = try
    using MPI
    MPI.Init()
    true
catch e
    @warn "MPI not available, running in single-GPU mode" exception=e
    false
end

using CUDA, Lux, Optimisers, LuxCUDA, Statistics, Random
using Lux: DistributedUtils
using TOML

# Only import NCCL if we're using distributed mode
if USE_DISTRIBUTED
    using NCCL
end

include("src/OptimizationProcedures/OptimizationProcedures.jl")
include("src/DatasetsModels/DatasetsModels.jl")
include("src/TrainArgs.jl")

using .DatasetsModels: toy_imagenet_data
using .OptimizationProcedures: lux_training!, resnet

# ═══════════════════════════════════════════════════════════════
# Setup: Distributed or Single-GPU
# ═══════════════════════════════════════════════════════════════
if USE_DISTRIBUTED
    # Initialize the backend (this sets up MPI internally)
    DistributedUtils.initialize(NCCLBackend)
    
    # Get the initialized backend
    const backend = DistributedUtils.get_distributed_backend(NCCLBackend)
    
    # Get rank information from the backend
    const global_rank = DistributedUtils.local_rank(backend)
    const world_size = DistributedUtils.total_workers(backend)
    
    # Set GPU device based on local rank
    const local_rank = global_rank % CUDA.ndevices()
    CUDA.device!(local_rank)
    
    println("[Rank $global_rank/$world_size] Using GPU $local_rank: $(CUDA.name(CUDA.device()))")
else
    const global_rank = 0
    const world_size = 1
    const backend = nothing
    
    # Use GPU if available, otherwise CPU
    if CUDA.functional()
        CUDA.device!(0)
        println("[Single-GPU Mode] Using GPU: $(CUDA.name(CUDA.device()))")
    else
        println("[Single-GPU Mode] No GPU available, using CPU")
    end
end

# ═══════════════════════════════════════════════════════════════
# Rest of your code stays the same!
# ═══════════════════════════════════════════════════════════════
cfg = TOML.parsefile("config.toml")
imagenet_path = cfg["paths"]["imagenet_path"]

model = resnet(; depth=18)
opt_state = Momentum(0.1f0, 0.9f0)

rng = Random.default_rng()
Random.seed!(rng, 1234)

# Backend is `nothing` in single-GPU mode, your toy_imagenet_data handles this
train_set, val_set, test_set = toy_imagenet_data(
    imagenet_path, 32, 32, 224;
    dev=gpu_device(), 
    backend=backend
)

println("[Rank $global_rank] Created $(length(train_set)) training batches")

# Initialize model parameters and synchronize across processes
ps, st = Lux.setup(rng, model) |> gpu_device()

if USE_DISTRIBUTED
    ps = DistributedUtils.synchronize!!(backend, ps)
    st = DistributedUtils.synchronize!!(backend, st)
end

# Wrap optimizer for distributed training
if USE_DISTRIBUTED
    opt_state = DistributedUtils.DistributedOptimizer(backend, opt_state)
end

# Setup optimizer state
tstate = Lux.Training.TrainState(model, ps, st, opt_state)

const vjp_method = AutoZygote()
const loss_function = Lux.CrossEntropyLoss(; logits=Val(true))

println("[Rank $global_rank] Starting training...")

for (i, batch) in enumerate(train_set)
    batch_time = time()

    global tstate
    
    if i ≤ 2
        imgs, lbls = batch
        println("[Rank $global_rank] Batch $i: imgsize=$(size(imgs)), lblsize=$(size(lbls))")
    end
    
    grads, loss, _, tstate = Lux.Training.compute_gradients(
        vjp_method, loss_function, batch, tstate)
    
    tstate = Lux.Training.apply_gradients!(tstate, grads)
    
    elapsed = time() - batch_time
    println("[Rank $global_rank] Batch $i: loss=$loss, time=$(elapsed)s")
    
    if i >= 5
        break
    end
end

println("[Rank $global_rank] Training complete")

if USE_DISTRIBUTED
    MPI.Finalize()
end