using Revise
include("HelperFunctions/loss_functions.jl")
include("HelperFunctions/lux_training.jl")
include("HelperFunctions/general_masked_model.jl")
include("HelperFunctions/tamade.jl")
include("HelperFunctions/assert_arg_correctness.jl")

"""
    procedure(
        train_set::Vector{<:Tuple},
        validation_set::Vector{<:Tuple},
        test_set::Vector{<:Tuple},
        tstate::Lux.Training.TrainState,
        loss_fun::LossFunction,
        args)::Tuple{Lux.Training.TrainState, Dict{String, Any}, LossFunction}

    This function runs a compression procedure. The specific procedure that is run depends on the loss_fun handed to procedure. The loss_fun is a struct and muliple dispatch then determines during training which optimization updates have to be run given this struct. The procedure function is usually not run directly. Instead, it is called by one of available procedures in this repo (e.g. DRR, PMMP or RL1).

    The procedure consists of 3 phases:
        - a regularized training phase,
        - a pruning phase,
        - and an unregularized finetuning phase, in which a mask fixes the pruned parameters.

    Returns tstate (containing all model, optimizer and network parameters), logs (containing loss and accuracy curves, execution times and other logged information) and loss_fun (the loss function that is determined by the procedure).

    Arguments:

        - `train_set`: The training set.
        - `validation_set`: The validation set.
        - `test_set`: The test set.
        - `tstate`: An object of type `Lux.Training.TrainState`, containing all model, optimizer and parameter information.
        - `loss_fun`: The regularized loss function (e.g. RL1_loss, DRR or PMMP)
        - `args`: The training arguments, a struct defined in the module `TrainingArguments`
"""
function procedure(
    train_set::Union{Vector{<:Tuple}, DeviceIterator},
    validation_set::Union{Vector{<:Tuple}, DeviceIterator},
    test_set::Union{Vector{<:Tuple}, DeviceIterator},
    tstate::Lux.Training.TrainState,
    loss_fun::LossFunction,
    args)::Tuple{Lux.Training.TrainState, Dict{String, Any}, LossFunction}

    assert_arg_correctness(args, validation_set)
    
    if args.verbose
        println("Training...")
    end

    # The training state is augmented by a mask that is later used during training to hold pruned parameters fixed.
    tstate = convert_to_general_masked_model!(tstate)
    if args.gauss_loss
        if !haskey(tstate.parameters, :sigma)
            tstate = Training.TrainState(tstate.model, (tstate.parameters..., sigma=args.dev([args.dtype(0.1)])), tstate.states, tstate.optimizer)
        end
    end

    # Phase 1 consists of a regularized training procedure.
    tstate, logs, loss_fun = lux_training!(train_set, validation_set, test_set, loss_fun, tstate, args; min_epochs=args.min_epochs, max_epochs=args.max_epochs, shrinking=args.shrinking, layerwise_pruning_flag=args.layerwise_pruning, converge_val_loss=args.converge_val_loss);

    # loss_fun_after_training = deepcopy(loss_fun)

    # Phase 1 consists of a pruning step.
    if args.layerwise_pruning
        lwp_input = [batch[1] for batch in train_set]
        tstate, layerwise_logs = layerwise_reverse_pruning(tstate, lwp_input; dtype = args.dtype, lr = args.layerwise_pruning_lr, alpha = args.layerwise_pruning_alpha, dev = args.dev, epochs=args.max_epochs, verbose=args.verbose, smoothing_window=args.smoothing_window, mask_start_value=args.layerwise_pruning_mask_start_value)
    end
    tstate, loss_fun = prune_and_shrink!(tstate, loss_fun, train_set, args.tolerated_relative_loss_increase, args.binary_search_resolution; dtype=args.dtype, dev=args.dev, delete_neurons=args.delete_neurons, random_gradient_pruning=args.random_gradient_pruning, final_epoch=true)

    if args.verbose
        println("\nTraining finished.\n\nFinetuning...")
    end
    
    # For the finetuning phase, unregularized optimization is performed, while employing the above defined mask to fix pruned parameters. To this end, the loss_fun is set to RL1_loss or RL1_Gauss with alpha=0, which is equivalent to unregularized loss.
    if args.gauss_loss
        finetuning_loss = RL1_Gauss(; alpha=args.dtype(0), rho=args.dtype(0), loss_f=loss_fun.loss_f) # RL1 Gauss with alpha=0=rho is like plain Gauss without L1 or L2 regularization and therefore appropriate here. Implementation is efficient.
    else
        finetuning_loss = RL1_loss(; alpha=args.dtype(0), rho=args.dtype(0), loss_f=loss_fun.loss_f)
    end
    
    # Phase 3 consists of finetuning.
    tstate, logs, loss_fun = lux_training!(train_set, validation_set, test_set, finetuning_loss, tstate, args; min_epochs=args.finetuning_min_epochs, max_epochs=args.finetuning_max_epochs, shrinking=args.finetuning_shrinking, layerwise_pruning_flag=args.finetuning_layerwise_pruning, converge_val_loss=args.finetuning_converge_val_loss)

    if args.verbose
        println("\nProcedure finished.\n")
    end

    return tstate, logs, loss_fun #, loss_fun_after_training
end