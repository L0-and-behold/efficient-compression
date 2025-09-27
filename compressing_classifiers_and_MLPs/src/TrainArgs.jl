"""
    module TrainingArguments

This module defines the `TrainArgs` type and its default values for training arguments.
    `args = TrainArgs()` is being used throughout the codebase as a configuration object.

# Usage

args = TrainArgs() # defaults to TrainArgs{Float32}()
or
args = TrainArgs{Float64}()
"""

abstract type AbstractTrainArgs end

using Optimisers: Adam, Descent
using LuxCUDA
using Lux: gpu_device, cpu_device

Base.@kwdef mutable struct TrainArgs{T<:Union{Float32,Float64}} <: AbstractTrainArgs
    architecture::Any = nothing
    optimization_procedure::Any = nothing
    dataset::Any = nothing
    architecture_teacher::Vector{Int} = [2,5,8,1]
    architecture_student::Vector{Int} = [2,25,25,1]
    dtype::DataType = T ###
    dev::Function = cpu_device() ###
    verbose::Bool = true
    optimizer = Adam
    lr::T = 1f-3
    min_epochs::Int = 10 # start pruning and convergence checks after this many epochs
    max_epochs::Int = 20000
    α::T = 1f1
    β::T = 5f0
    ρ::T = 0f0
    L1_alpha::T = 0f0
    tolerated_relative_loss_increase::T = 0.01
    NORM::Bool = false
    layer_NORM::Bool = true
    train_set_size::Union{String, Int} = 300
    train_batch_size::Union{String, Int} = 300
    val_set_size::Union{String, Int} = 1000
    val_batch_size::Union{String, Int} = 1000
    test_set_size::Union{String, Int} = 5000
    test_batch_size::Union{String, Int} = 5000
    noise::T = 0.8f-1
    gauss_loss::Bool = true
    random_gradient_pruning::Bool = true
    shrinking::Bool = false
    delete_neurons::Bool = false
    binary_search_resolution::T = 1f-7
    prune_window::Int = 10 # pruning and convergence checks are only done every prune_window epochs (after min_epochs elapsed)
    shrinking_from_deviation_of::T = 1f-2
    smoothing_window::Int = 30
    finetuning_shrinking::Bool = false
    finetuning_min_epochs::Int = smoothing_window+3 # this should be >= (smoothing_window + 3) to ensure proper convergence.
    finetuning_max_epochs::Int = 10000
    finetuning_layerwise_pruning::Bool = false
    layerwise_pruning::Bool = false
    layerwise_pruning_alpha::T = 1f-1
    layerwise_pruning_lr::T = 1f-3
    layerwise_pruning_mask_start_value::T = 0.3
    log_val_loss::Bool = true
    converge_val_loss::Bool = true
    finetuning_converge_val_loss::Bool = true
    logs = nothing
    multiply_mask_after_each_batch::Bool = true
    initial_p_value::T = 1.0
    initial_u_value::T = 0.0
    u_value_multiply_factor::T = 1.0
    gradient_repetition_factor::Int64 = 1
    seed::Int = 42
end

function TrainArgs(; T=Float32)
    return TrainArgs{T}()
end
