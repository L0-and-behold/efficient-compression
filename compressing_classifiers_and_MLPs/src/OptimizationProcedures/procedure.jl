"""
    procedure(
    train_set::Any,
    validation_set::Any,
    test_set::Any,
    tstate::Lux.Training.TrainState,
    loss_fun::LossFunction,
    args::AbstractTrainArgs,
    checkpoint::CheckpointManager
    )::Tuple{Lux.Training.TrainState, Dict{String, Any}, LossFunction, CheckpointManager}

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

using .Checkpointer

function procedure(
    train_set::Any,
    validation_set::Any,
    test_set::Any,
    tstate::Lux.Training.TrainState,
    loss_fun::LossFunction,
    args::AbstractTrainArgs,
    checkpoint::CheckpointManager
    )::Tuple{Lux.Training.TrainState, Dict{String, Any}, LossFunction, CheckpointManager}

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

    # Phase 1 consists of a regularized training procedure. Checkpointing during this stage. No checkpointing afterwards.
    tstate, logs, loss_fun, checkpoint = lux_training!(train_set, validation_set, test_set, loss_fun, tstate, args, checkpoint; min_epochs=args.min_epochs, max_epochs=args.max_epochs, shrinking=args.shrinking, layerwise_pruning_flag=args.layerwise_pruning, converge_val_loss=args.converge_val_loss, checkpoint_enabled=true);

    # Phase 2 consists of a pruning step.
    if args.layerwise_pruning
        lwp_input = [batch[1] for batch in train_set]
        tstate, layerwise_logs = layerwise_reverse_pruning(tstate, lwp_input; dtype = args.dtype, lr = args.layerwise_pruning_lr, alpha = args.layerwise_pruning_alpha, dev = args.dev, epochs=args.max_epochs, verbose=args.verbose, smoothing_window=args.smoothing_window, mask_start_value=args.layerwise_pruning_mask_start_value)
    end

    # Report state before pruning
    if args.verbose
        if haskey(logs, "validation_accuracy") && !isempty(logs["validation_accuracy"])
            println("  Pre-pruning val accuracy: $(round(logs["validation_accuracy"][end] * 100; digits=2))%")
        end
        if haskey(tstate.states, :mask)
            l0_before = recursive_sum(tstate.states.mask, args.dtype(0))
            println("  Pre-pruning L0 norm: $(Int(round(l0_before)))")
        end
    end

    if args.save_pre_pruning_model
        args.logs["pre_pruning_tstate"] = tstate
    end

    tamade_data = isnothing(args.tamade_calibration_batches) ? validation_set : Iterators.take(validation_set, args.tamade_calibration_batches)
    tstate, loss_fun = prune_and_shrink!(tstate, loss_fun, tamade_data, args.tolerated_relative_loss_increase, args.binary_search_resolution; dtype=args.dtype, dev=args.dev, delete_neurons=args.delete_neurons, random_gradient_pruning=args.random_gradient_pruning, final_epoch=true, val_acc_tolerance=args.tamade_val_acc_tolerance)

    # Report sparsity and accuracy after pruning
    if args.verbose
        if haskey(tstate.states, :mask)
            l0_after = recursive_sum(tstate.states.mask, args.dtype(0))
            println("  Post-pruning L0 norm: $(Int(round(l0_after)))")
        end
        post_prune_acc = accuracy(tstate, validation_set)
        println("  Post-pruning val accuracy: $(round(post_prune_acc * 100; digits=2))%")
    end

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
    tstate, logs, loss_fun, checkpoint = lux_training!(train_set, validation_set, test_set, finetuning_loss, tstate, args, checkpoint; min_epochs=args.finetuning_min_epochs, max_epochs=args.finetuning_max_epochs, shrinking=args.finetuning_shrinking, layerwise_pruning_flag=args.finetuning_layerwise_pruning, converge_val_loss=args.finetuning_converge_val_loss, checkpoint_enabled=false)

    if args.verbose
        if haskey(logs, "validation_accuracy") && !isempty(logs["validation_accuracy"])
            println("  Post-FT val accuracy: $(round(logs["validation_accuracy"][end] * 100; digits=2))%")
        end
        println("\nProcedure finished.\n")
    end

    # Ensure pruned weights are exactly zero in the saved model
    if haskey(tstate.states, :mask)
        recursively_multiply!(tstate.parameters.p, tstate.states.mask)
    end

    return tstate, logs, loss_fun, checkpoint #, loss_fun_after_training
end