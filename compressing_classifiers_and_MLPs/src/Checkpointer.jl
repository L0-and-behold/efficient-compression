module Checkpointer

using JLD2, Dates, UUIDs, CUDA, Lux
using Base: @kwdef

using ..TrainingArguments: TrainArgs, AbstractTrainArgs

export CheckpointMetadata, CheckpointContent, CheckpointManager, maybe_load_checkpoint, maybe_save_checkpoint, load_checkpoint, save_checkpoint, find_available_checkpoint, list_checkpoint_status, clean_checkpoint_files, should_stop_for_timeout, save_metadata

@kwdef mutable struct CheckpointMetadata
    checkpoint_id::String  = string(uuid4()) # Unique ID for this run
    status::Symbol = :running # :available, :running, :finished
    type::Symbol = :fresh_run # :fresh_run, loaded_run
    created_at::Float64 = time()
    last_updated::Float64 = time()
    max_runtime_seconds::Union{Nothing, Float64} = nothing
    start_time::Float64 = time()
    path::String=""
end

@kwdef mutable struct CheckpointContent
    args::AbstractTrainArgs=TrainArgs{Float32}()
    tstate=nothing
    epoch::Int=1
    prev_val_loss::Number = 0f0
    prev_prev_val_loss::Number = 0f0
    best_tstate = nothing
    convergence_triggered::Bool = false
end
mutable struct CheckpointManager
    do_checkpointing::Bool
    metadata::CheckpointMetadata
    content::CheckpointContent
end

function save_metadata(filepath::String, metadata::CheckpointMetadata)
    jldopen(filepath, "r+") do f
        if haskey(f, "metadata")
            delete!(f, "metadata")
        end
        f["metadata"] = metadata
    end
end

function maybe_save_checkpoint(checkpoint::CheckpointManager)
    if checkpoint.do_checkpointing
        save_checkpoint(checkpoint.metadata, checkpoint.content)
    end
end
function save_checkpoint(metadata::CheckpointMetadata, content::CheckpointContent)

    for field in fieldnames(CheckpointContent)
        if getfield(content, field) == nothing 
            @warn "CheckpointContent field `$(field)` is `nothing`."
        end
    end

    mkpath(metadata.path)
    path = joinpath(metadata.path, metadata.checkpoint_id*".jld2")
    jldopen(path, "w") do f
        f["metadata"] = metadata
        f["content"]  = content
    end
end

function find_available_checkpoint(checkpoint_dir::String)::Union{String, Nothing}
    if !isdir(checkpoint_dir)
        @warn "Path does not exist: $checkpoint_dir"
        return nothing
    end

    for file in readdir(checkpoint_dir)
        if endswith(file, ".jld2")
            filepath = joinpath(checkpoint_dir, file)
            try
                @load filepath metadata
                @assert metadata.status isa Symbol
                if metadata.status == :available
                    return filepath
                elseif metadata.status != :running && metadata.status != :finished
                    @warn "Encountered checkpoint with unknown status: $(metadata.status)"
                end
            catch e
                @warn "Could not read checkpoint $file: $e"
            end
        end
    end
    return nothing
end

function list_checkpoint_status(checkpoint_dir::String)
    if !isdir(checkpoint_dir)
        println("Directory not found: $checkpoint_dir")
    end

    for file in readdir(checkpoint_dir)
        if endswith(file, ".jld2")
            filepath = joinpath(checkpoint_dir, file)
            try
                @load filepath metadata
                println("Checkpoint: $(metadata.checkpoint_id), status: $(metadata.status), type: $(typeof(metadata.status)) ")
            catch e
                println("Could not read checkpoint $file. Rethrowing error:")
                rethrow(e)
            end
        end
    end
end

function maybe_load_checkpoint(checkpoint::CheckpointManager, args::AbstractTrainArgs)::Tuple{CheckpointManager, Bool}
    # No checkpointing requested → nothing loaded.
    if !(checkpoint.do_checkpointing)
        return checkpoint, false
    end

    filepath = find_available_checkpoint(checkpoint.metadata.path)
    if filepath !== nothing
        # Load metadata, update its status, then persist the change before loading heavy content.
        metadata, content = load_checkpoint(filepath, args)
        checkpoint.metadata = metadata
        checkpoint.content = content
        return checkpoint, true
    else
        return checkpoint, false
    end
end

function load_checkpoint(filepath::String, args::AbstractTrainArgs)::Tuple{CheckpointMetadata, CheckpointContent}
    # Load metadata only
    metadata = jldopen(filepath, "r") do f
        f["metadata"]
    end

    # Update status fields
    metadata.status = :running
    metadata.type = :loaded_run
    metadata.start_time = time()
    metadata.last_updated = time()
    save_metadata(filepath, metadata)

    # Load the heavy content after metadata has been saved
    content = jldopen(filepath, "r") do f
        f["content"]
    end
    
    if content.best_tstate == nothing && content.tstate != nothing
        @warn "Checkpoint loaded with a `nothing` best_tstate. Substituting for tstate."
        content.best_tstate = deepcopy(content.tstate)
    end
    
    content.tstate |> args.dev
    content.best_tstate |> args.dev

    return metadata, content
end

function clean_checkpoint_files(checkpoint_dir::String)
    if !isdir(checkpoint_dir)
        error("Path does not exist: $checkpoint_dir")
    end
    for file in readdir(checkpoint_dir)
        if endswith(file, ".jld2")
            filepath = joinpath(checkpoint_dir, file)
            try
                @load filepath metadata
                @assert metadata.status isa Symbol
                if metadata.status == :available || metadata.status == :running
                    continue
                elseif metadata.status == :finished
                    rm(filepath)
                    println("Removed finished experiment checkpoint $(metadata.checkpoint_id)")
                else
                    error("Found checkpoint with disallowed status: $metadata.status")
                end
            catch e
                println("Warning: Could not read checkpoint $file: $e")
            end
        end
    end
end

function should_stop_for_timeout(metadata::CheckpointMetadata)::Bool
    elapsed = time() - metadata.start_time
    # Stop if 90% of max runtime reached (buffer for saving)
    return elapsed >= 0.90 * metadata.max_runtime_seconds
end

end # module Checkpointer