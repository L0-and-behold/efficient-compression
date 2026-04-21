module Checkpointer

using JLD2, Dates, UUIDs, CUDA, Lux
using Base: @kwdef

using ..TrainingArguments: TrainArgs, AbstractTrainArgs

export CheckpointMetadata, CheckpointContent, CheckpointManager,
       generate_checkpoint_id,
       maybe_save_checkpoint, save_checkpoint,
       load_checkpoint, load_checkpoint_by_id,
       mark_checkpoint_finished!,
       find_available_checkpoint, list_checkpoint_status, clean_checkpoint_files

const _ADJECTIVES = ["brave","calm","dark","eager","fierce","gentle","happy","icy",
                     "jolly","keen","lively","merry","noble","proud","quiet","rapid",
                     "swift","tame","wild","bold","amber","crisp","dusty","fleet",
                     "grand","hollow","iron","jade","lunar","misty","olive","plush",
                     "ruddy","stark","thorny","umber","vivid","wry","zesty","ashen"]
const _ANIMALS    = ["bear","crane","deer","eagle","fox","goat","hawk","ibis",
                     "jaguar","koala","lynx","moose","otter","panda","quail","raven",
                     "seal","tiger","viper","wolf","bison","condor","dingo","finch",
                     "gecko","heron","impala","jackal","kestrel","lemur","marmot",
                     "narwhal","osprey","python","quokka","rhino","shrike","tapir","urial"]

generate_checkpoint_id() = "$(rand(_ADJECTIVES))-$(rand(_ANIMALS))"

@kwdef mutable struct CheckpointMetadata
    checkpoint_id::String = generate_checkpoint_id()
    type::Symbol = :fresh_run  # :fresh_run, :loaded_run
    created_at::Float64 = time()
    last_updated::Float64 = time()
    path::String = ""
end

@kwdef mutable struct CheckpointContent
    args::AbstractTrainArgs = TrainArgs{Float32}()
    tstate = nothing
    epoch::Int = 1
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

function maybe_save_checkpoint(checkpoint::CheckpointManager)
    if checkpoint.do_checkpointing
        save_checkpoint(checkpoint.metadata, checkpoint.content)
    end
end

function save_checkpoint(metadata::CheckpointMetadata, content::CheckpointContent)
    for field in fieldnames(CheckpointContent)
        if getfield(content, field) == nothing && field != :args  # args fields checked separately
            @warn "CheckpointContent field `$(field)` is `nothing`."
        end
    end
    mkpath(metadata.path)
    path = joinpath(metadata.path, metadata.checkpoint_id * ".jld2")
    # Strip closures/lambdas that JLD2 cannot serialize by name.
    # Both are restored from the caller's args on resume.
    content_to_save = deepcopy(content)
    content_to_save.args.dataset   = nothing
    content_to_save.args.optimizer = nothing
    jldopen(path, "w") do f
        f["metadata"] = metadata
        f["content"]  = content_to_save
    end
end

function load_checkpoint(filepath::String, args::AbstractTrainArgs)::Tuple{CheckpointMetadata, CheckpointContent}
    metadata = jldopen(filepath, "r") do f
        f["metadata"]
    end
    metadata.type = :loaded_run

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

function load_checkpoint_by_id(dir::String, id::String, args::AbstractTrainArgs)::Tuple{CheckpointMetadata, CheckpointContent}
    filepath = joinpath(dir, "$id.jld2")
    isfile(filepath) || error("No checkpoint found for id='$id' at $filepath")
    return load_checkpoint(filepath, args)
end

function mark_checkpoint_finished!(checkpoint::CheckpointManager)
    id  = checkpoint.metadata.checkpoint_id
    src = joinpath(checkpoint.metadata.path, "$id.jld2")
    dst = joinpath(checkpoint.metadata.path, "FINISHED_$id.jld2")
    if isfile(src)
        mv(src, dst; force=true)
        println("  Checkpoint $id → FINISHED")
    end
end

# Utility: scan directory for any non-FINISHED checkpoint (for inspection)
function find_available_checkpoint(checkpoint_dir::String)::Union{String, Nothing}
    if !isdir(checkpoint_dir)
        @warn "Path does not exist: $checkpoint_dir"
        return nothing
    end
    for file in readdir(checkpoint_dir)
        if endswith(file, ".jld2") && !startswith(file, "FINISHED_")
            return joinpath(checkpoint_dir, file)
        end
    end
    return nothing
end

function list_checkpoint_status(checkpoint_dir::String)
    if !isdir(checkpoint_dir)
        println("Directory not found: $checkpoint_dir")
        return
    end
    for file in readdir(checkpoint_dir)
        if endswith(file, ".jld2")
            status = startswith(file, "FINISHED_") ? "finished" : "active"
            println("$file  [$status]")
        end
    end
end

function clean_checkpoint_files(checkpoint_dir::String)
    if !isdir(checkpoint_dir)
        error("Path does not exist: $checkpoint_dir")
    end
    for file in readdir(checkpoint_dir)
        if endswith(file, ".jld2") && startswith(file, "FINISHED_")
            filepath = joinpath(checkpoint_dir, file)
            rm(filepath)
            println("Removed finished checkpoint: $file")
        end
    end
end

end # module Checkpointer
