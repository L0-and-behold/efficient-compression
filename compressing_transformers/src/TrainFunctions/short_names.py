from collections import Counter

def get_short_names(args):
    short_names = {}

    short_names["run_id"] = "RunID"
    short_names["python_config_file"] = "PyCF"
    
    ### Regularization Parameters - BEGIN
    short_names["alpha"] = "A"                          # Regularization strength for ℓ₀-Regularization
    short_names["initial_p_value"] = "P"                 # Initial p value for PMMP method
    short_names["initial_u_value"] = "U"                 # Initial u value for PMMP method
    short_names["beta"] = "B"                           # Sharpness parameter β for DRR method
    ### Regularization Parameters - END

    ### Model Configuration - BEGIN 
    short_names["training_procedure"] = "TP"       # Training procedure to use (rl1, vanilla, drr, or pmmp)
    short_names["transformer_config"] = "TC"    # Transformer config
    ### Model Configuration - END

    ### Optimizer Parameters - BEGIN
    short_names["learning_rate"] = "LR"                 # Initial learning rate for optimizer
    short_names["AdamW_betas"] = "AdB"                   # The beta parameters for the AdamW optimizer (typical choices include (0.9, 0.95) or (0.9, 0.999))
    short_names["warmup_steps"] = "WS"                         # Warmup increases the learning rate linearly to `learning_rate` in `warmup_steps` steps
    short_names["weight_decay"] = "WD"                         # Weight decay applies L2 regularization to all parameters except biases, LayerNorm-weights and `u` and `p` parameters of PMMP
    ### Optimizer Parameters - END

    ### Training Process Parameters - BEGIN
    short_names["iterations_per_epoch"] = "IpE"              # the number of iterations or batches which are processed per epoch. Each iteration, `batch_size` many `seq_length` long batches are processed. Therefore, make sure the dataset contains at least `args["iterations_per_epoch"]*batch_size*seq_length` many tokens (where `seq_length` is the context window of the model).
    short_names["tokens_per_epoch"] = "TpE"             # (False or int). If not False, then ["iterations_per_epoch"] is overwritten and set equal to int(args["tokens_per_epoch"] / batch_size / seq_length), where seq_length is the context window of the model. Make sure args["tokens_per_epoch"] is not bigger than the number of tokens in your dataset. This parameter can also be used to make the training token number per epoch equal to N times the number of parameters of the model.
    short_names["epochs_prelude"] = "EP"                    # Number of epochs for prelude phase (unregularized)
    short_names["epochs"] = "E"                            # Number of epochs for main training
    short_names["epochs_fine_tuning"] = "EF"                # Number of epochs for fine-tuning phase (unregularized with smaller model)
    short_names["stop_epoch_at_batch_prelude"] = "SEP"   # Whether to stop prelude epoch early at batch n (False or int)
    short_names["stop_epoch_at_batch"] = "SE"           # Whether to stop main training epoch early at batch n (False or int)
    short_names["stop_epoch_at_batch_fine_tuning"] = "SEF" # Whether to stop fine-tuning epoch early at batch n (False or int)
    ### Training Process Parameters - END


    ### Pruning Parameters - BEGIN
    short_names["do_pruning"] = "Pr"                     # Whether to perform model pruning
    short_names["first_pruning_after"] = "FPr"               # Epoch after which to start pruning
    short_names["prune_every"] = "PrE"                       # Prune model every n epochs
    ### Pruning Parameters - END


    ### Model Loading Parameters - BEGIN
    short_names["use_pretrained_model"] = "PreM"          # Whether to use a pretrained model
    short_names["use_model_from_experiment"] = "ExpM"      # Name of experiment to load model from (None or str)
    short_names["use_model_from_run"] = "IDM"             # Run ID to load model from (None or str)
    short_names["elapsed_epochs"] = "ElE"                    # Total epochs to be recorded (must match expected total if continuing training)
    ### Model Loading Parameters - END

    ### General Training Parameters - BEGIN
    short_names["seed"] = "Se"                              # Random seed for reproducibility
    short_names["tolerated_relative_loss_increase"] = "Tol" # Maximum tolerated loss increase during TAMADE
    short_names["steps_per_chunk"] = "SpC"                    # Number of optimization steps per data chunk
    short_names["log_every"] = "LogFreq"                           # Log metrics every n optimization steps
    short_names["checkpoint_time"] = "CpT"                 # Save checkpoint every n seconds
    short_names["max_runtime"] = "MaxR"                     # Maximum runtime in seconds before forced termination
    ### General Training Parameters - END

    short_names["only_process_every_nth_batch_when_calculating_train_loss"] = "OpNTr"      # Only every args["only_process_every_nth_batch_when_calculating_train_loss"] of the train set is evaluated for computing the train loss. Numbers bigger 1 increase speed but make the estimate less accurate.
    short_names["only_process_every_nth_batch_when_calculating_test_loss"] = "OpNTe"      # Only every args["only_process_every_nth_batch_when_calculating_test_loss"] of the test set is evaluated for computing the test loss. Numbers bigger 1 increase speed but make the estimate less accurate.
    ### Metrics Calculation Parameters - END

    ### Check for duplicate shortnames and if the short_names keys coincide with the args keys - BEGIN

    # Make sure there are no duplicate short names
    values = list(short_names.values())
    counts = Counter(values)
    duplicates = {v: c for v, c in counts.items() if c > 1}
    assert not duplicates, f"Duplicate short names found: {duplicates}"

    # Make sure the keys in args and short_names coincide
    assert set(args.keys()) == set(short_names.keys()), \
    f"Mismatch: {set(args.keys()) ^ set(short_names.keys())}" # display the symmetric difference if they do not coincide

    ### Check for duplicate shortnames and if the short_names keys coincide with the args keys - END

    return short_names