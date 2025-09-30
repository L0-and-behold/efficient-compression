include("procedure.jl")

import MLUtils: DeviceIterator
function recursively_modify_FPP!(params, fun)
    for subparams in params
        if isa(subparams, AbstractArray{T} where T)
            subparams .= fun(subparams)
        elseif isa(subparams, NamedTuple) && !isempty(subparams)
            recursively_modify_FPP!(subparams, fun)
        end
    end
end

function convert_tstate!(tstate, args)
    pw = deepcopy(tstate.parameters)
    pp = deepcopy(tstate.parameters)
    u = deepcopy(tstate.parameters)
    
    recursively_modify_FPP!(pp, x -> args.initial_p_value .* one.(x))
    recursively_modify_FPP!(u, x -> args.initial_u_value .* one.(x))

    tstate = Training.TrainState(tstate.model, (p = tstate.parameters, pw=pw, pp=pp, u=u), tstate.states, tstate.optimizer)
    if hasproperty(tstate.model, :name)
        @reset tstate.model.name = "FPP model"
    elseif hasproperty(tstate.model.layer, :name)
        @reset tstate.model.layer.name = "FPP model"
    end
    return tstate
end

"""
    FPP_procedure(
        train_set::Union{Vector{<:Tuple}, DeviceIterator},
        validation_set::Union{Vector{<:Tuple}, DeviceIterator},
        test_set::Union{Vector{<:Tuple}, DeviceIterator},
        tstate::Lux.Training.TrainState,
        loss_fctn::Function,
        args)::Tuple{Lux.Training.TrainState, Dict{String, Any}, LossFunction}
    
    This function runs a FPP compression procedure. During this procedure, an L0 norm term is added to a given objective and a minimax optimization process computes the gradients of a probabilistic reformulation of this L0 norm augmented objective. 

    During optimization, weights are additionally pruned with random gradient pruning and binary search based threshold pruning. The final training phase consists of a finetuning phase, in which the regularization is relaxed by withdrawing the L0 regularization but a mask ensures that already pruned weights are not updated anymore.

    Returns a procedure call, which in turn returns tstate (containing all model, optimizer and network parameters), logs (containing loss and accuracy curves, execution times and other logged information) and loss_fun (the loss function that is determined by the procedure).

    Arguments:

        - `train_set`: The training set.
        - `validation_set`: The validation set.
        - `test_set`: The test set.
        - `tstate`: An object of type `Lux.Training.TrainState`, containing all model, optimizer and parameter information.
        - `loss_fctn`: The unregularized loss function (e.g. logitcrossentropy or MSELoss)
        - `args`: The training arguments, a struct defined in the module `TrainingArguments`
"""
function FPP_procedure(
    train_set::Union{Vector{<:Tuple}, DeviceIterator},
    validation_set::Union{Vector{<:Tuple}, DeviceIterator},
    test_set::Union{Vector{<:Tuple}, DeviceIterator},
    tstate::Lux.Training.TrainState,
    loss_fctn::Function,
    args)::Tuple{Lux.Training.TrainState, Dict{String, Any}, LossFunction}
    
    if !haskey(tstate.parameters, :pp)
        tstate = convert_tstate!(tstate, args)
    end
    if !((@isdefined loss_fun) && typeof(loss_fun) <: FPP)
        initial_grad_p = deepcopy(tstate.parameters.p)
        recursively_modify_FPP!(initial_grad_p, x -> zero.(x))

        grad_template = (p = initial_grad_p, pw = deepcopy(initial_grad_p), pp = deepcopy(initial_grad_p), u = deepcopy(initial_grad_p))

        parameter_avgs = deepcopy(tstate.parameters)
        
        model_param_number = args.dtype(Lux.parameterlength(tstate.parameters.p))
        grad_template |> args.dev
        parameter_avgs |> args.dev

        if args.gauss_loss
            sigma = [0.1f0] |> args.dev
            grad_template = (grad_template..., sigma=sigma) |> args.dev
            loss_fun = FPP_Gauss(grad_template, parameter_avgs, model_param_number;  gradient_repetition_factor=args.gradient_repetition_factor, alpha=args.α, rho=args.ρ, u_value_multiply_factor=args.u_value_multiply_factor, loss_f=loss_fctn, L1_alpha=args.L1_alpha)
        else
            loss_fun = FPP(grad_template, parameter_avgs, model_param_number; gradient_repetition_factor=args.gradient_repetition_factor, alpha=args.α, rho=args.ρ, u_value_multiply_factor=args.u_value_multiply_factor, loss_f=loss_fctn, L1_alpha=args.L1_alpha)
        end
    end

    return procedure(train_set, validation_set, test_set, tstate, loss_fun, args)
end
