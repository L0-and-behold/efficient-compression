"""
Collection of the different optimization procedures.

An optimization procedure is a function with
    Input:
     - train_set
        - a vector of tuples (x, y). x is an input batch tensor, y is a target batch tensor.
     - validation_set
        - a vector of a single tuple (x, y). x is an input batch tensor, y is a target batch tensor. The validation set is prepared as a single batch.
     - tstate
        - a Lux.TrainState
     - loss_fctn
        - a loss function such as:
            Lux.MSELoss()
    Output:
     - tstate
        - Lux.TrainState
     - logs
        - a dictionary with information on loss
            - "epochs" : the epoch at which the following were calculated
            - "train_loss"
            - "validation_loss"
    Behavior:
    - tstate is modified
    - Calculation on CPU or GPU, depending on args.dev (CPU is much faster for smaller datasets and models like the teacher-student models but GPU is much faster for bigger models like the VGG classifier)

"""

module OptimizationProcedures
   using Lux, Optimisers, PlotlyJS, CUDA, Revise, JLD2

   using ..Checkpointer
   using ..TrainingArguments
   
   include("HelperFunctions/loss_functions.jl")
   export accuracy

   # include("LayerWiseFunctions/")
   include("HelperFunctions/lux_training.jl")
   include("HelperFunctions/general_masked_model.jl")
   export testmode_states
   include("HelperFunctions/tamade.jl")
   include("HelperFunctions/assert_arg_correctness.jl")
   include("HelperFunctions/named_tuple_operations.jl")
   include("HelperFunctions/delete_neurons.jl")
   include("HelperFunctions/recursive_operations.jl")
   include("HelperFunctions/update_functions.jl")
   include("LayerWiseFunctions/projected_implementation/layerwise_pruning.jl")
   include("HelperFunctions/random_gradient_pruning.jl")


   # __precompile__()

   include("HelperFunctions/generate_networks_and_data.jl")
   export generate_dense_network, generate_dataset, setup_data_teacher_and_student, generate_tstate, scale_alpha_rho!, Lenet_MLP, Lenet_5, Lenet_5_Caffe, VGG, resnet50, resnet18, toy_resnet, alexnet

   include("HelperFunctions/plot_networks_and_their_output.jl")
   export plot_weights, net_scatter_plot, net_smooth_plot, plot_data_teacher_and_student

   include("HelperFunctions/break_loop.jl")
   export is_saturated

   include("procedure.jl")
   export procedure

   include("PMMP_procedure.jl")
   export PMMP_procedure

   include("FPP_procedure.jl")
   export FPP_procedure

   include("DRR_procedure.jl")
   export DRR_procedure, get_layer_number, get_block_number

   include("RL1_procedure.jl")
   export RL1_procedure

   include("layerwise_procedure.jl")
   export layerwise_procedure
   
end # module