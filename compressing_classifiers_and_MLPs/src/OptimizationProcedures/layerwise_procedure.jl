using Revise, CUDA
import Lux
include("procedure.jl")

import MLUtils: DeviceIterator

"""
    layerwise_procedure(
        train_set::Union{Vector{<:Tuple}, DeviceIterator},
        validation_set::Union{Vector{<:Tuple}, DeviceIterator},
        test_set::Union{Vector{<:Tuple}, DeviceIterator},
        tstate::Lux.Training.TrainState,
        loss_fctn::Function,
        args)::Tuple{Lux.Training.TrainState, Dict{String, Any}, LossFunction}
    
    This function runs a layerwise compression procedure. During this procedure, the network is pruned layer by layer every args.prune_window epochs. The pruning in a given layer is performed, employing a probabilistic reformulation of L0 regularized regression. 

    During optimization, weights are additionally pruned with random gradient pruning and binary search based threshold pruning. The final training phase consists of a finetuning phase, where a mask ensures that already pruned weights are not updated anymore. Whether layerwise pruning is also performed during finetuning depeds on whether args.finetuning_layerwise_pruning is set to true or false.

    Returns a procedure call, which in turn returns tstate (containing all model, optimizer and network parameters), logs (containing loss and accuracy curves, execution times and other logged information) and loss_fun (the loss function that is determined by the procedure).

    Arguments:

        - `train_set`: The training set.
        - `validation_set`: The validation set.
        - `test_set`: The test set.
        - `tstate`: An object of type `Lux.Training.TrainState`, containing all model, optimizer and parameter information.
        - `loss_fctn`: The unregularized loss function (e.g. logitcrossentropy or MSELoss)
        - `args`: The training arguments, a struct defined in the module `TrainingArguments`
"""
function layerwise_procedure(
    train_set::Union{Vector{<:Tuple}, DeviceIterator},
    validation_set::Union{Vector{<:Tuple}, DeviceIterator},
    test_set::Union{Vector{<:Tuple}, DeviceIterator},
    tstate::Lux.Training.TrainState,
    loss_fctn::Function,
    args)::Tuple{Lux.Training.TrainState, Dict{String, Any}, LossFunction}
    
    if args.gauss_loss
        loss_fun = RL1_Gauss(; alpha=args.dtype(0), rho=args.dtype(0), loss_f=loss_fctn)
    else
        loss_fun = RL1_loss(; alpha=args.dtype(0), rho=args.dtype(0), loss_f=loss_fctn)
    end

    args.layerwise_pruning = true

    return procedure(train_set, validation_set, test_set, tstate, loss_fun, args)
end