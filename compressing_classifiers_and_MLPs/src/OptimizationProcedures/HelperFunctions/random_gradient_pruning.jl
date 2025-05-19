using Zygote, Lux, LuxCUDA, Optimisers, Revise
import Lux

"""
    random_gradient(tstate, data_size, loss_fun; dev=cpu_device(), vjp=AutoZygote())

    This function takes in a tstate (containing all model, optimizer and network parameters of a neural network), a data (batch) size and a loss function, samples a random data batch and thereafter returns the gradients with respect to the loss function applied to that tstate and random batch.
"""
function random_gradient(tstate, data_size, loss_fun; dev=cpu_device(), vjp=AutoZygote())
    r_data = (randn(Float32, data_size[1]),randn(Float32, data_size[2])) |> dev
    
    m, ps, st = tstate.model, tstate.parameters, tstate.states

    if !(loss_fun isa GenericLossFunction)
        return Training.compute_gradients(vjp, loss_fun, r_data, tstate)[1]
    end

    if haskey(st, :layers)
        loss_wrapper = ps -> begin
            y_pred = m(r_data[1], ps, st)[1]
            loss_fun(y_pred, r_data[2])
        end
    elseif haskey(st, :st)
        apply_chain = (x, ps, st) -> begin
            for (layer, layer_ps, layer_st) in zip(m.layers, ps.p, st.st)
                x, _ = Lux.apply(layer, x, layer_ps, layer_st)
            end
            return x
        end
        loss_wrapper = ps -> begin
            y_pred = apply_chain(r_data[1], ps, st)
            loss_fun(y_pred, r_data[2])
        end
    else
        error("Unknown state structure")
    end

    return Zygote.gradient(loss_wrapper, ps)[1]
end

"""
    prune_with_random_gradient!(tstate, data_size, loss_fun; dev=cpu_device())

    This function takes in a tstate (containing all model, optimizer and network parameters of a neural network) and a loss function, calls the random gradient function to obtain random gradients with respect to this loss function, and then prunes weights in the network by setting all those weights to zero whose random gradient is exactly equal to zero.
    
    It does not return anything but is a function with side effects that modifies tstate.
"""
function prune_with_random_gradient!(tstate, data_size, loss_fun; dev=cpu_device())
    rg = random_gradient(tstate, data_size, loss_fun; dev=dev)
    if haskey(tstate.parameters,:p)
        zero_non_contributing_weights!(tstate.parameters.p, rg.p)
    else
        zero_non_contributing_weights!(tstate.parameters, rg)
    end
end

function zero_non_contributing_weights!(ps::NamedTuple, r_grads::NamedTuple)
    for (l1, l2) in zip(ps, r_grads)
        if isa(l1, NamedTuple) && isa(l2, NamedTuple)
            zero_non_contributing_weights!(l1, l2)
        else
            if !isnothing(l2)
                l1[l2 .== 0] .= 0
            end
        end
    end
end