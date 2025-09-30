
using Revise
include("procedure.jl")

"""
    RL1_procedure(
        train_set::Union{Vector{<:Tuple}, DeviceIterator},
        validation_set::Union{Vector{<:Tuple}, DeviceIterator},
        test_set::Union{Vector{<:Tuple}, DeviceIterator},
        tstate::Lux.Training.TrainState,
        loss_fctn::Function,
        args)::Tuple{Lux.Training.TrainState, Dict{String, Any}, LossFunction}
    
    This function runs a relaxed L1-regularized compression procedure. During this procedure, a given loss function is augmented with an L1-norm regularization term and then optimized.
    
    During optimization, weights are additionally pruned with random gradient pruning and binary search based threshold pruning. The final training phase consists of a finetuning phase, in which the regularization is relaxed (hence Relaxed-L1) by withdrawing the L1 regularization but a mask ensures that already pruned weights are not updated anymore.

    Returns a procedure call, which in turn returns tstate (containing all model, optimizer and network parameters), logs (containing loss and accuracy curves, execution times and other logged information) and loss_fun (the loss function that is determined by the procedure).

    Arguments:

        - `train_set`: The training set.
        - `validation_set`: The validation set.
        - `test_set`: The test set.
        - `tstate`: An object of type `Lux.Training.TrainState`, containing all model, optimizer and parameter information.
        - `loss_fctn`: The unregularized loss function (e.g. logitcrossentropy or MSELoss)
        - `args`: The training arguments, a struct defined in the module `TrainingArguments`
"""
function RL1_procedure(
    train_set::Union{Vector{<:Tuple}, DeviceIterator},
    validation_set::Union{Vector{<:Tuple}, DeviceIterator},
    test_set::Union{Vector{<:Tuple}, DeviceIterator},
    tstate::Lux.Training.TrainState,
    loss_fctn::Function,
    args)::Tuple{Lux.Training.TrainState, Dict{String, Any}, LossFunction}
    
    if args.gauss_loss
        if hasproperty(tstate.model, :name)
            comparison_name = tstate.model.name
        elseif hasproperty(tstate.model, :layer)
            if hasproperty(tstate.model.layer, :name)
                comparison_name = tstate.model.layer.name
            else
                comparison_name = ""
            end
        else
            comparison_name = ""
        end
        @assert comparison_name == "teacher-student network"
        loss_fun = RL1_Gauss(; alpha=args.α, rho=args.ρ, loss_f=loss_fctn)
    else
        loss_fun = RL1_loss(; alpha=args.α, rho=args.ρ, loss_f=loss_fctn)
    end

    return procedure(train_set, validation_set, test_set, tstate, loss_fun, args)
end