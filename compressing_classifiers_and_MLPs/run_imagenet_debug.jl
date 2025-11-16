using Pkg
Pkg.activate(".")

using Distributed
# If the script is started with `julia -p N` the workers already exist.
# Otherwise start them manually (for debugging):
if nworkers() == 1      # only the master process
    # change the number below to the number of GPUs you have on the node
    @info "Launching 4 workers for DDP (adjust to your GPU count)…"
    addprocs(3; exeflags="--project=$(Base.active_project())")
end

# Broadcast all needed packages to every worker
@everywhere begin
    using Pkg
    Pkg.activate(".")
    using Suppressor
end
@everywhere begin
    @suppress begin
        include("src/OptimizationProcedures/OptimizationProcedures.jl")
    end
    include("src/DatasetsModels/DatasetsModels.jl")
    include("src/TrainArgs.jl")
end
@everywhere begin
    using CUDA, MPI, NCCL, Lux, Optimisers, LuxCUDA, Statistics, Random
    using Lux: DistributedUtils
    using .DatasetsModels: imagenet_data, toy_imagenet_data
    using .OptimizationProcedures: lux_training!, resnet
end

#   (rank 1 → GPU 0, rank 2 → GPU 1, …)
@everywhere begin
    local_rank = myid() - 1   # `myid()==2` is the first worker after the master
    if local_rank ≥ 0
        # CUDA.device!(local_rank)
        CUDA.device!(0) # give every worker the same device (on compsrv)
        @info "Worker $(myid()) → CUDA device $(CUDA.device())"
    end
end

# Normal imports that are only needed on the master (for printing)
using TOML
using Random
using Revise

function print_gpu_info(msg::String)
    device = CUDA.device()
    println("[GPU INFO] $msg – device: $device (", CUDA.name(device), ")")
end

cfg = TOML.parsefile("config.toml")
imagenet_path = cfg["paths"]["imagenet_path"]
println("ImageNet path: $imagenet_path")

# Create model and training state (master creates the *template* model)
println("Creating model (ResNet‑18)…")
model = resnet(; depth=18)

opt_state = Momentum(0.1f0, 0.9f0)
master_rng = Random.default_rng()
Random.seed!(master_rng, 1234)

# -----------------------------------------------------------------
# Initialize DDP backend on each worker
# -----------------------------------------------------------------
@everywhere begin
    # Calculate rank: worker 2 → rank 0, worker 3 → rank 1, etc.
    local_rank = myid() - 2
    total_workers = nworkers()
    
    # Initialize the NCCL backend with proper rank information
    # backend = DistributedUtils.NCCLBackend(local_rank, total_workers)
    backend = nothing
    
    @info "Worker $(myid()) initialized with rank=$local_rank of $total_workers"
end

# -----------------------------------------------------------------
# Each worker independently creates its own data loaders with the backend
# For Imagenet we will need to change this a bit. Here, each worker get their own dataloader without the dataloader of the others in mind.
# -----------------------------------------------------------------
@everywhere begin
    rng = deepcopy($master_rng)
    
    # Each worker creates its own data loaders with the distributed backend
    # The DistributedDataContainer will automatically shard the data
    train_set, val_set, test_set = toy_imagenet_data(
        $imagenet_path, 32, 32, 224;
        dev=gpu_device(), 
        backend=backend  # Pass the initialized backend
    )
    @info "Worker $(myid()) created data loaders: $(length(train_set)) training batches"
    @info "Worker $(myid()) – $(length(train_set)) batches"
end

batch_counts = [ @fetchfrom w length(train_set) for w in workers() ]
global_batches = sum(batch_counts)

# -----------------------------------------------------------------
# Initialize model and optimizer state on each worker
# -----------------------------------------------------------------
@everywhere begin
    ps, st = Lux.setup(rng, $model) |> Lux.gpu_device()
    global tstate = Lux.Training.TrainState($model, ps, st, $opt_state)
end

println("# training batches per worker: ", @fetchfrom 2 length(train_set))
print_gpu_info("After data loading")

# -----------------------------------------------------------------
# Run training loop on each worker independently
# -----------------------------------------------------------------
println("Starting DDP training…")

@everywhere begin
    const vjp_method = AutoZygote()
    const loss_function = Lux.CrossEntropyLoss(; logits=Val(true))
end


# Each worker runs its training loop independently
# The gradient synchronization happens automatically via the backend
@sync begin
    for w in workers()
        @async @fetchfrom w begin
            local_rank = myid() - 2
            println("[Worker $(myid()), Rank $local_rank] Starting training loop on GPU $(CUDA.device())")
            
            for (i, batch) in enumerate(train_set)
                global tstate
                batch_time = time()

                if i ≤ 2
                    imgs, lbls = batch
                    @info "Worker $(myid()) – batch $i: " *
                            "imgsize=$(size(imgs))  lblsize=$(size(lbls))"
                end
        
                # Compute gradients locally
                grads, loss, _, tstate = Training.compute_gradients(
                    vjp, loss_fun, batch, tstate)

                # Apply gradients - this should trigger all-reduce via the backend
                # NOTE: You may need to manually call the all-reduce depending on 
                # your Lux version. Check Lux.DistributedUtils documentation.
                tstate = Training.apply_gradients!(tstate, grads)
                
                # Synchronize gradients across workers (if not done automatically)
                # ps = DistributedUtils.synchronize!!(backend, tstate.parameters)
                # tstate = Lux.Training.TrainState(tstate.model, ps, tstate.states, tstate.optimizer_state)

                elapsed = time() - batch_time
                println("[Worker $(myid()), Rank $local_rank] Batch $i: loss=$(loss), time=$(elapsed)s")
                
                # Optional: break after a few batches for testing
                if i >= 5
                    break
                end
            end
            
            println("[Worker $(myid()), Rank $local_rank] Training complete")
        end
    end
end

println("DDP training finished.")
print_gpu_info("End of script")