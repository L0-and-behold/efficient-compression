using Lux, LuxCUDA, Optimisers, Printf, Random, Zygote, Accessors, Revise

include("break_loop.jl")
include("tamade.jl")
include("update_functions.jl")
include("../LayerWiseFunctions/projected_implementation/layerwise_pruning.jl")

function recursive_sum(mask, l0_mask)
    for m in mask
        if isa(m, NamedTuple) && !isempty(m)
            l0_mask = recursive_sum(m, l0_mask)
        elseif isa(m, AbstractArray)
            l0_mask += sum(m)
        end
    end
    return l0_mask
end

function saturation_test(a;b=0.1,c=0.9)
    if a <= b || a >= c
        return true
    else
        return false
    end
end

function recursive_comparison_violation_count(nt, comparator, tot_count=0)
    for m in nt
        if isa(m, NamedTuple) && !isempty(m)
            tot_count = recursive_comparison_violation_count(m, comparator, tot_count)
        elseif isa(m, AbstractArray)
            tot_count += count(.!comparator.(m))
        end
    end
    return tot_count
end

function recursive_comparison(A, comparator, val) # for example: recursive_comparison(tstate.parameters.pp, <=, 1.0) == true
    for a in A
        if isa(a, NamedTuple) && !isempty(a)
            if !recursive_comparison(a, comparator, val)
                return false
            end
        elseif isa(a, AbstractArray)
            if any(.!comparator.(a,val))
                return false
            end
        end
    end
    return true
end

"""
    lux_training!(train_set, validation_set, test_set, loss_fun, tstate, args; vjp = AutoZygote(), min_epochs = 20000, max_epochs=40000, shrinking=true, layerwise_pruning_flag=false, converge_val_loss=true)

    This function is a training routine for neural networks. Each step of the training consists of an update phase, a pruning phase and a convergence check phase.
    
    In the update phase, it hands batches of data, the loss function and the tstate (containing the neural network model and parameters) to an update function in update_functions.jl. The loss function is a julia struct and the update function determines, based on the properties of this struct, which update to perform.
    
    In the pruning phase, which is only run every args.prune_window steps and only after min_epochs, layerwise pruning is performed if layerwise_pruning_flag=true and a prune function is called if shrinking=true.
    
    In the convergence check phase, which is also only run every args.prune_window steps and only after min_epochs, convergence is triggered either if the validation loss is saturated or, if no validation set is provided, if the train loss is saturated. Saturation, in turn, is computed by the is_saturated function in break_loop.jl.

    Returns tstate (containing all model, optimizer and network parameters), logs (containing loss and accuracy curves, execution times and other logged information) and loss_fun (the loss function that is determined by the procedure).

    Arguments:

        - `train_set`: The training set.
        - `validation_set`: The validation set.
        - `test_set`: The test set.
        - `loss_fun`: The regularized loss function (e.g. RL1_loss, DRR or PMMP)
        - `tstate`: An object of type `Lux.Training.TrainState`, containing all model, optimizer and parameter information.
        - `args`: The training arguments, a struct defined in the module `TrainingArguments`
        - `vjp`: The auto-differentiation backend (default is AutoZygote())
        - `min_epochs`: The minimum number of traning epochs before which no convergence is checked and before which no pruning takes place (default is 20000 but should be changed depending on problem class)
        - `max_epochs`: The maximum number of traning epochs. (default is 40000)
        - `shrinking`: A boolean that determines whether regular pruning every args.prune_window epochs is carried out or not in the pruning phase of the training. (default is true)
        - `layerwise_pruning_flag`: A boolean that determines whether layerwise pruning is carried out every args.prune_window epochs.
        - `converge_val_loss`: A boolean that determines whether the validation loss or the train loss should be used to determine convergence. (default is true, which however requires to hand a non-empty validation set to lux_training!)
"""
function lux_training!(train_set, validation_set, test_set, loss_fun, tstate, args; vjp = AutoZygote(), min_epochs = 20000, max_epochs=40000, shrinking=true, layerwise_pruning_flag=false, converge_val_loss=true)
    
    convergence_triggered = false
    if args.log_val_loss
        log_arg = "val_loss"
    else
        log_arg = "train_loss"
    end
    if converge_val_loss && args.log_val_loss
        conv_arg = "val_loss"
    else
        conv_arg = "train_loss"
    end
    if layerwise_pruning_flag
        input = [batch[1] for batch in train_set]
    end
    # Initialization
    start_epoch = 0
    if isnothing(args.logs)
        args.logs = Dict{String, Any}(
            "epochs" => Int[],
            "train_loss" => args.dtype[],
            "val_loss" => args.dtype[],
            "test_loss" => Tuple{Int, Number}[],
            "epoch_execution_time"  => args.dtype[],
            "total_time" => args.dtype(0),
            "sigmas" => args.dtype[],
            "final_train_accuracy" => args.dtype(0),
            "validation_accuracy" => args.dtype[],
            "test_accuracy" => Tuple{Int, Number}[],
            "test_loss" => Tuple{Int, Number}[],
            "l0_mask" => args.dtype[],
            "converged_at" => Int[],
            "turning_points_val_loss" => Int[],
            "best_tstate_points" => Int[]
        )
    else
        @assert isa(args.logs, Dict{String, Any})
        for (key, value) in [("epochs", Int[]), ("train_loss", args.dtype[]), ("val_loss", args.dtype[]), ("epoch_execution_time", args.dtype[]), ("total_time", args.dtype(0)), ("sigmas" => args.dtype[]), ("accuracy" => args.dtype[]), ("l0_mask" => args.dtype[]), ("test_loss" => Tuple{Int, Number}[]), ("validation_accuracy" => args.dtype[]), ("test_accuracy" => Tuple{Int, Number}[]), ("test_loss" => Tuple{Int, Number}[]), ("final_train_accuracy" => args.dtype(0)), ("converged_at" => Int[]), ("turning_points_val_loss" => Int[]), ("best_tstate_points" => Int[])]
            if !haskey(args.logs, key)
                args.logs[key] = value
            end
        end
        if haskey(args.logs, "epochs")
            if !isempty(args.logs["epochs"])
                start_epoch = args.logs["epochs"][end]
            end
        end
    end
    num_batches = args.dtype(length(train_set))

    # Main loop
    total_time_start = time()
    prev_val_loss = Inf32
    prev_prev_val_loss = 0
    best_tstate = deepcopy(tstate)
    obs_window = round(Int,args.smoothing_window/1)
    start_turn_point = length(args.logs["turning_points_val_loss"])
    for epoch in 1:max_epochs
        
        # Batch loop
        epoch_loss = zero(args.dtype)
        epoch_start_time = time()
        for batch in train_set
            tstate, loss, stats = update_state!(vjp, loss_fun, batch, tstate)
            if haskey(tstate.states, :mask) && args.multiply_mask_after_each_batch
                recursively_multiply!(tstate.parameters.p, tstate.states.mask)
            end
            epoch_loss += loss
        end
        epoch_end_time = time()
        epoch_loss /= num_batches

        
        push!(args.logs["epochs"], start_epoch+epoch)
        push!(args.logs["train_loss"], epoch_loss)
        push!(args.logs["epoch_execution_time"], epoch_end_time - epoch_start_time)
        
        if haskey(tstate.states, :mask)
            l0_mask= recursive_sum(tstate.states.mask, args.dtype(0))
            push!(args.logs["l0_mask"], l0_mask)
        end
        if args.log_val_loss
            epoch_val_loss = zero(args.dtype)
            for val_batch in validation_set
                epoch_val_loss += loss_fun(tstate.model, tstate.parameters, tstate.states, val_batch)[1]
            end
            epoch_val_loss /= length(validation_set)
            push!(args.logs["val_loss"], epoch_val_loss)
            if args.verbose 
                @printf "\rEpoch: %5d \t Train_loss: %.4g \t Val_loss: %.4g \t" epoch epoch_loss epoch_val_loss
            end

            if !isnothing(test_set)
                epoch_test_loss = zero(args.dtype)
                for test_batch in test_set
                    epoch_test_loss += loss_fun(tstate.model, tstate.parameters, tstate.states, test_batch)[1]
                end
                epoch_test_loss /= length(test_set)
                push!(args.logs["test_loss"], (start_epoch+epoch, epoch_test_loss))
            end            
        end
        if args.verbose && !args.log_val_loss
            @printf "\rEpoch: %5d \t Train_loss: %.4g \t" epoch epoch_loss
        end
        if haskey(tstate.parameters, :sigma)
            push!(args.logs["sigmas"], sum(tstate.parameters.sigma))
        end
        if loss_fun.loss_f == logitcrossentropy
            push!(args.logs["validation_accuracy"], accuracy(tstate, validation_set))
        end
        
        # Pruning and Convergence check is done at most every prune_window epochs
        if epoch % args.prune_window == 0 || epoch == max_epochs

            if converge_val_loss && args.log_val_loss
                if prev_val_loss <= epoch_val_loss
                    if haskey(tstate.parameters, :pp) # additionally check for active set convergence
                        violation_count = recursive_comparison_violation_count(tstate.parameters.pp, saturation_test)
                        if violation_count / Lux.parameterlength(tstate.model) <= 0.01 # percentage of violations lower than threshold
                            push!(args.logs["turning_points_val_loss"], start_epoch+epoch)
                        end
                    else
                        push!(args.logs["turning_points_val_loss"], start_epoch+epoch)
                    end
                    comparison_losses = args.logs["val_loss"][args.logs["turning_points_val_loss"][start_turn_point+1:end]]
                    if (!isempty(comparison_losses)) && (epoch_val_loss <= minimum(comparison_losses))
                        best_tstate = deepcopy(tstate)
                        push!(args.logs["best_tstate_points"], start_epoch+epoch)
                    end
                end
                prev_prev_val_loss = prev_val_loss
                prev_val_loss = epoch_val_loss
            end
            if epoch >= max(min_epochs, obs_window + 3)
                
                if loss_fun.loss_f == logitcrossentropy
                    push!(args.logs["test_accuracy"], (epoch, accuracy(tstate, test_set)))
                end
                
                if shrinking && (is_saturated(args.logs[conv_arg], args.smoothing_window; min_possible_deviation=args.shrinking_from_deviation_of)  || epoch == max_epochs)

                    if layerwise_pruning_flag
                        tstate, layerwise_logs = layerwise_reverse_pruning(tstate, input; dtype = args.dtype, lr = args.layerwise_pruning_lr, alpha = args.layerwise_pruning_alpha, dev = args.dev, epochs=args.max_epochs, verbose=args.verbose, smoothing_window=args.smoothing_window, mask_start_value=args.layerwise_pruning_mask_start_value)
                    end

                    if args.verbose
                        println("\nShrinking")
                    end

                    if epoch == max_epochs
                        final_epoch=true
                    else
                        final_epoch=false
                    end
                    tstate, _, loss_fun = prune_and_shrink!(tstate, nothing, loss_fun, train_set, args.tolerated_relative_loss_increase, args.binary_search_resolution; dtype=args.dtype, dev=args.dev, delete_neurons=args.delete_neurons, random_gradient_pruning=args.random_gradient_pruning, final_epoch=final_epoch)
                end

                convergence_condition = is_saturated(args.logs[conv_arg], args.smoothing_window)

                if haskey(tstate.parameters, :pp)
                    violation_count = recursive_comparison_violation_count(tstate.parameters.pp, saturation_test)
                    if violation_count / Lux.parameterlength(tstate.model) <= 0.01 # percentage of violations lower than threshold
                        pp_conv_cond = true
                    else
                        pp_conv_cond = false
                    end
                else
                    pp_conv_cond = true
                end
                if convergence_condition && pp_conv_cond
                    push!(args.logs["converged_at"], epoch)

                    @printf "\nConverged. Epoch: %3d \t Loss: %.5g\n" epoch args.logs[conv_arg][end]
                    convergence_triggered = true
                    break
                end
            end
        end
    end
    total_time_end = time()
    args.logs["total_time"] += total_time_end - total_time_start

    args.logs["final_train_accuracy"] = accuracy(tstate, train_set)
    if !convergence_triggered
        push!(args.logs["converged_at"], max_epochs)
    end

    # take the tstate closest to where val_loss was smallest:
    if converge_val_loss && args.log_val_loss
        comparison_losses = args.logs["val_loss"][args.logs["turning_points_val_loss"][start_turn_point+1:end]]
        if isempty(comparison_losses) || args.logs["val_loss"][end] <= minimum(comparison_losses)
            return_tstate = tstate
        else
            return_tstate = best_tstate
            println("returned best tstate with val loss ", minimum(comparison_losses))
        end
    else
        return_tstate = tstate
    end

    return return_tstate, args.logs, loss_fun
end
