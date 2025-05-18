using Accessors, Revise

function recursively_modify!(params, fun)
    if !isnothing(params)
        for subparams in params
            if isa(subparams, AbstractArray{T} where T)
                subparams .= fun(subparams)
            else
                recursively_modify!(subparams, fun)
            end
        end
    end
end

"""
    convert_to_general_masked_model!(tstate)

    Takes a tstate (containing all model, optimizer and network parameters), loss_fun (the loss function) and returns a new tstate that is augmented by a mask, which can then be used during training to prevent backpropagation on masked weights.
"""
function convert_to_general_masked_model!(tstate)
    if !haskey(tstate.parameters, :p)
        tstate = Training.TrainState(tstate.model, (p=tstate.parameters,), tstate.states, tstate.optimizer)
    end
    if !haskey(tstate.states, :mask) || (haskey(tstate.states, :mask) && isnothing(tstate.states.mask))
        mask = deepcopy(tstate.parameters.p)
        recursively_modify!(mask, x -> one.(x))
        @reset tstate.states = (st = tstate.states, mask = mask)
    end
    return tstate
end