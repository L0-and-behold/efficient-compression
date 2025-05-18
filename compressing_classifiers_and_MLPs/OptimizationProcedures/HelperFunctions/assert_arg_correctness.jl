using Revise
function assert_arg_correctness(args, validation_set)
    # @assert length(validation_set) == 1 "Validation set must contain exactly one batch"
    
    @assert args.α >= 0.0 "got $(args.α)"
    @assert args.β >= 0.0 "got $(args.β)"
    @assert args.tolerated_relative_loss_increase >= 0.0 "got $(args.tolerated_relative_loss_increase)"
    @assert args.lr > 0.0 "got $(args.lr)"
    @assert args.min_epochs > 0 "got $(args.min_epochs)"
    @assert args.max_epochs > 0 "got $(args.max_epochs)"
    @assert args.max_epochs >= args.min_epochs "got $(args.max_epochs) < $(args.min_epochs)"
    return
end