include("named_tuple_operations.jl")
include("delete_neurons.jl")
include("random_gradient_pruning.jl")

using Accessors, Revise

"""
    TAMADE(f::function, target::Float64, low::Float64, high::Float64, tol::Float64, binary_search_resolution=1e-7)

    Computes the highest threshold `thr` such that `f(thr)` is still lower or equal to `(1+tol)*target`.

    Returns thr, steps, corresponding to the highest allowed threshold and the number of steps until convergence, respectively.

    Arguments:
        - `f`: A (loss) function that takes in a threshold number and outputs another number quantifying the loss. It must be monotone enough for binary search to be applicable.
        - `target`: The target to which `f(thr)` is compared. Is often the original loss.
        - `low`: The initial low point for binary search. Is often 0.
        - `high`: The initial high point for binary search. Is often equal to 2 times the maximum absolute weight value.
        - `tol`: The allowed tolerance. Recall that `f(thr)` is compared to `(1+tol)*target` during the binary search update step.
        - `binary_search_resolution`: The maximum allowed resolution for thr. This means that thr is only updated until the binary search interval becomes smaller than `binary_search_resolution`.
"""
function TAMADE(f, target, low, high, tol, binary_search_resolution=1e-7)
    thr = low
    steps = 0
    
    while abs(high - low) > binary_search_resolution # this determines the precision of the thr value
        steps += 1
        mid = (low + high) / 2
        if (low+high)/2 == low # this can become true if numerical precision becomes insufficient. Therefore, this is performed as a security check to prevent falling into an infinite loop caused by numerical precision issues
            return thr, steps
        end
        if f(mid) <= (1+tol)*target  # no abs(f(mid) - target) in this line - it is sufficient if f(mid)-target <= tol*target because we "search_for_highest_x_such_that_...". Furthermore, the multiplication of tol and target means that tol acts as a percentage: One requires that f(mid) <= (1+tol)*target, and since target is the initial loss, this requires that the pruned network does not have a loss that exceeds the initial loss by tol*100 percent. This makes the choice of tol less dependent on the specific model and data.
            thr = mid
            low = mid  # Move the lower bound up to find the highest x
        # elseif f(mid) < target # if f and target and tol are always >=0 and initial low = 0, then this never happens. Otherwise uncomment.
        #     low = mid
        else
            high = mid
        end
    end
    return thr, steps  # Return the highest x that meets the condition
end

function recursive_prune!(p, thr)
    for sp in p
        if isa(sp, NamedTuple) && !isempty(sp)
            recursive_prune!(sp, thr)
        elseif isa(sp, AbstractArray)
            sp[abs.(sp) .< thr] .= 0
        end
    end
end
function recursive_copy!(p1, p2)
    for (sp1, sp2) in zip(p1, p2)
        if isa(sp1, NamedTuple) && !isempty(sp1)
            recursive_copy!(sp1, sp2)
        elseif isa(sp1, AbstractArray)
            sp1 .= sp2
        end
    end
end


function compute_loss_over_batches(tstate, params, data, dtype, loss_fctn)
    total_loss = zero(dtype)
    for batch in data
        total_loss += loss_fctn(tstate.model, params, tstate.states, batch)[1]
    end
    total_loss = total_loss / dtype(length(data))
    return total_loss
end


function find_right_pruning_threshold(tstate, loss_fun, data, tolerance, binary_search_resolution=1e-7; dtype=Float32, prune_input=false)
    # this function performs binary search until the simplest model is found, such that abs(loss_with_pruning - loss_without_pruning) <= tolerance
   
    if haskey(tstate.parameters, :p)
        ps2 = (p=deepcopy(tstate.parameters.p),)
    else
        ps2 = (p=deepcopy(tstate.parameters),)
    end
    if haskey(tstate.parameters, :sigma) # even when Loss is gauss, do binary search with MSELoss
        loss_fctn = RL1_loss(; alpha = 0f0, rho = 0f0, loss_f = loss_fun.loss_f)
    else
        loss_fctn = loss_fun
    end
    loss_without_pruning = compute_loss_over_batches(tstate, tstate.parameters, data, dtype, loss_fctn)

    # As first step, compute the first threshold as the maximum of the absolute values of the weights:
    initial_thr = zero(dtype)
    function recursive_get_max!(p)
        for sp in p
            if isa(sp, NamedTuple) && !isempty(sp)
                recursive_get_max!(sp)
            elseif isa(sp, AbstractArray)
                initial_thr = max(initial_thr, maximum(abs.(sp)))
            end
        end
    end
    if haskey(tstate.parameters, :p)
        recursive_get_max!(tstate.parameters.p)
    else
        recursive_get_max!(tstate.parameters)
    end
    
    # next start the binary search with thr as initial guess:
    function get_loss(thr)
        if haskey(tstate.parameters, :p)
            recursive_copy!(ps2.p, tstate.parameters.p)
            recursive_prune!(ps2.p, thr)
            return compute_loss_over_batches(tstate, ps2, data, dtype, loss_fctn)
        else
            recursive_copy!(ps2.p, tstate.parameters)
            recursive_prune!(ps2.p, thr)
            return compute_loss_over_batches(tstate, ps2.p, data, dtype, loss_fctn)
        end
    end
    
    optimal_thr, steps = TAMADE(get_loss, loss_without_pruning, zero(dtype), 2*initial_thr, tolerance, binary_search_resolution)
    
    if prune_input
        if haskey(tstate.parameters,:p)
            recursive_prune!(tstate.parameters.p, optimal_thr)
        else
            recursive_prune!(tstate.parameters, optimal_thr)
        end
    end
    return tstate, optimal_thr, steps, initial_thr
end

"""
    prune_and_shrink!(tstate, loss_fun, data, tolerance, binary_search_resolution=1e-7 ; dtype=Float32, dev=cpu_device(), delete_neurons=true, random_gradient_pruning=true, final_epoch=false)

    Function that prunes tstate (containing all model, optimizer and network parameters of a neural network) by calling prune_with_random_gradient! and find_right_pruning_threshold (which in turn calls the binary search routine TAMADE).

    It also updates mask parameters to ensure that pruned parameters stay zero. Finally it handles the compatibility of the ordinary mask and the parameter mask of the PMMP procedure.

    Arguments:
        - `tstate`: The Lux TrainState
        - `loss_fun`: The loss function determined by the optimization procedure
        - `data`: The dataset
        - `tolerance`: The tolerance for the TAMADE procedure: The loss is allowed to increase by at most `tol*100` percent.
        - `binary_search_resolution`: The pruning threshold is only determined up to `binary_search_resolution`.
        - `dtype`: The datatype (usually Float32)
        - `dev`: The device (either CPU or GPU)
        - `delete_neurons`: Boolean that determines whether neurons are supposed to be deleted or not.
        - `random_gradient_pruning`: Boolean that determines whether random gradient pruning should be performed to eliminate spurious weights or not.
        - `final_epoch`: A Boolean that indicates whether training reached its final epoch or not. This is important for the PMMP procedure where pruning should only be performed in the last epoch to prevent interference with the PMMP mask.
"""
function prune_and_shrink!(tstate, loss_fun, data, tolerance, binary_search_resolution=1e-7 ; dtype=Float32, dev=cpu_device(), delete_neurons=true, random_gradient_pruning=true, final_epoch=false)
    
    if haskey(tstate.states, :mask)
        recursively_set_to_zero!(tstate.parameters.p, tstate.states.mask)
    end

    if random_gradient_pruning
        data_size = [size(data[1][1]),size(data[1][2])]
        prune_with_random_gradient!(tstate, data_size, loss_fun; dev=dev)
    end

    if haskey(tstate.parameters, :pp)
        recursively_set_to_zero!(tstate.parameters.pp, tstate.parameters.p)
    end
    if haskey(tstate.parameters, :pp) && !final_epoch
        if haskey(tstate.states, :mask)
            recursively_set_to_zero!(tstate.states.mask, tstate.parameters.pp)
        end
    else
        println("\nPruning...")
        tstate, optimal_thr, steps, initial_thr = find_right_pruning_threshold(tstate, loss_fun, data, tolerance, binary_search_resolution; dtype=dtype, prune_input=true)
    end

    if random_gradient_pruning
        data_size = [size(data[1][1]),size(data[1][2])]
        prune_with_random_gradient!(tstate, data_size, loss_fun; dev=dev)
    end

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
    if delete_neurons && (comparison_name == "teacher-student network" || comparison_name == "PMMP model" || comparison_name == "masked model")
        tstate, loss_fun  = delete_neurons_lux!(tstate, loss_fun; reassign=true)
    elseif delete_neurons && comparison_name != "teacher-student network"
        println("Warning: args.delete_neurons == true but comparison_name != 'teacher-student network'. Neuron deletion is currently only implemented for teacher-student networks.")
    end

    if haskey(tstate.states, :mask)
        recursively_set_to_zero!(tstate.states.mask, tstate.parameters.p)
    end
    if haskey(tstate.parameters, :pp)
        recursively_set_to_zero!(tstate.parameters.pp, tstate.parameters.p)
    end
    
    return tstate, loss_fun
end

function recursively_set_to_zero!(p1, p2)
    for (sp1, sp2) in zip(p1, p2)
        if isa(sp1, NamedTuple) && !isempty(sp1)
            recursively_set_to_zero!(sp1, sp2)
        elseif isa(sp1, AbstractArray) && !isempty(sp1)
            sp1[sp2 .== 0] .= 0
        end
    end
end