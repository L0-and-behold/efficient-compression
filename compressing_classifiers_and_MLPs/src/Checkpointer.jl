using JLD2, Dates

struct CheckpointMetadata
    checkpoint_id::String          # Unique ID for this run
    status::Symbol                 # :available, :running, :completed
    created_at::DateTime
    last_updated::DateTime
    current_epoch::Int
    max_runtime_seconds::Union{Nothing, Float64}
    start_time::Float64
end

function save_checkpoint(checkpoint_dir::String, checkpoint_id::String, tstate, args, epoch::Int, prev_val_loss::Number, 
                        best_tstate, loss_fun::Function, convergence_triggered::Bool, metadata)
    checkpoint_path = joinpath(checkpoint_dir, "checkpoint_$(checkpoint_id).jld2")
    metadata_updated = CheckpointMetadata(
        checkpoint_id,
        :running,
        metadata.created_at,
        now(),
        epoch,
        metadata.max_runtime_seconds,
        metadata.start_time
    )
    
    @save checkpoint_path tstate args epoch prev_val_loss best_tstate loss_fun convergence_triggered metadata_updated
    println("✓ Checkpoint saved at epoch $epoch")
end

function finalize_checkpoint(checkpoint_dir::String, checkpoint_id::String)
    checkpoint_path = joinpath(checkpoint_dir, "checkpoint_$(checkpoint_id).jld2")
    if isfile(checkpoint_path)
        @load checkpoint_path tstate args epoch prev_val_loss best_tstate loss_fun convergence_triggered metadata
        metadata_final = CheckpointMetadata(
            metadata.checkpoint_id,
            :completed,
            metadata.created_at,
            now(),
            metadata.current_epoch,
            metadata.max_runtime_seconds,
            metadata.start_time
        )
        @save checkpoint_path tstate args epoch prev_val_loss best_tstate loss_fun convergence_triggered metadata_final
        println("✓ Checkpoint marked as completed")
    end
end

function find_available_checkpoint(checkpoint_dir::String)::Union{String, Nothing}
    if !isdir(checkpoint_dir)
        return nothing
    end
    
    for file in readdir(checkpoint_dir)
        if endswith(file, ".jld2")
            filepath = joinpath(checkpoint_dir, file)
            try
                @load filepath metadata
                if metadata.status == :available
                    return filepath
                end
            catch e
                println("Warning: Could not read checkpoint $file: $e")
            end
        end
    end
    return nothing
end

function should_stop_for_timeout(metadata)::Bool
    if isnothing(metadata.max_runtime_seconds)
        return false
    end
    
    elapsed = time() - metadata.start_time
    # Stop if 95% of max runtime reached (buffer for saving)
    return elapsed >= 0.95 * metadata.max_runtime_seconds
end