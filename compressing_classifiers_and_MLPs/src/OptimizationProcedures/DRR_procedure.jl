using Revise, CUDA
import Lux
include("procedure.jl")

"""
    DRR_procedure(
        train_set::Union{Vector{<:Tuple}, DeviceIterator},
        validation_set::Union{Vector{<:Tuple}, DeviceIterator},
        test_set::Union{Vector{<:Tuple}, DeviceIterator},
        tstate::Lux.Training.TrainState,
        loss_fctn::Function,
        args)::Tuple{Lux.Training.TrainState, Dict{String, Any}, LossFunction}
    
    This function runs a DRR compression procedure. During this procedure, a given objective is augmented with a smooth approximation of the L0 norm that can then be optimized via backpropagation. 

    During optimization, weights are additionally pruned with random gradient pruning and binary search based threshold pruning. The final training phase consists of a finetuning phase, in which the regularization is relaxed by withdrawing the smooth approximation of the L0 regularization but a mask ensures that already pruned weights are not updated anymore.

    Returns a procedure call, which in turn returns tstate (containing all model, optimizer and network parameters), logs (containing loss and accuracy curves, execution times and other logged information) and loss_fun (the loss function that is determined by the procedure).

    Arguments:

        - `train_set`: The training set.
        - `validation_set`: The validation set.
        - `test_set`: The test set.
        - `tstate`: An object of type `Lux.Training.TrainState`, containing all model, optimizer and parameter information.
        - `loss_fctn`: The unregularized loss function (e.g. logitcrossentropy or MSELoss)
        - `args`: The training arguments, a struct defined in the module `TrainingArguments`
"""
function DRR_procedure(
    train_set::Union{Vector{<:Tuple}, DeviceIterator},
    validation_set::Union{Vector{<:Tuple}, DeviceIterator},
    test_set::Union{Vector{<:Tuple}, DeviceIterator},
    tstate::Lux.Training.TrainState,
    loss_fctn::Function,
    args)::Tuple{Lux.Training.TrainState, Dict{String, Any}, LossFunction}
    
    if args.gauss_loss
        @assert tstate.model.name == "teacher-student network"
        loss_fun = DRR_Gauss(; NORM=args.NORM, alpha=args.α, beta=args.β, rho=args.ρ, loss_f=loss_fctn)
    else    
        model_param_number = args.dtype(Lux.parameterlength(tstate.parameters))
        if args.NORM
            if args.layer_NORM
                layernumber_model = get_layer_number(tstate.parameters)
                fun1 = DRR_NORM_layer_modification
            else
                layernumber_model = get_block_number(tstate.parameters)
                fun1 = DRR_NORM_modification
            end
        else
            layernumber_model = get_layer_number(tstate.parameters)
            fun1 = DRR_modification
        end
        loss_fun = DRR(model_param_number, layernumber_model; alpha=args.α, beta=args.β, rho=args.ρ, loss_f=loss_fctn, fun1 = fun1)
    end

    return procedure(train_set, validation_set, test_set, tstate, loss_fun, args)
end

function get_layer_number(params)
    i = 0
    function recursively_get_layers!(params)
        if !isnothing(params)
            for (name,subparam) in pairs(params)
                if startswith(string(name), "layer_") && !isempty(subparam)
                    i += 1
                elseif isa(subparam, NamedTuple) && !isempty(subparam)
                    recursively_get_layers!(subparam)
                end
            end
        end
    end
    recursively_get_layers!(params)
    return i
end
function get_block_number(params)
    i = 0
    function recursively_get_blocks!(params)
        if !isnothing(params)
            for (name,subparam) in pairs(params)
                if startswith(string(name), "weight") 
                    i += 1
                elseif isa(subparam, NamedTuple) && !isempty(subparam)
                    recursively_get_blocks!(subparam)
                end
            end
        end
    end
    recursively_get_blocks!(params)
    return i
end