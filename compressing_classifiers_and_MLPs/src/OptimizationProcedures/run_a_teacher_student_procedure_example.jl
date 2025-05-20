"""
A simple script to run, test and develop the different optimization procedures in the Teacher-Student Setup.
"""

using Revise
using CUDA
include("../TrainArgs.jl")
include("HelperFunctions/generate_networks_and_data.jl")
include("HelperFunctions/plot_networks_and_their_output.jl")

# Load the different L0_regularization procedures
include("RL1_procedure.jl")
include("DRR_procedure.jl")
include("PMMP_procedure.jl")
include("layerwise_procedure.jl")
# Note that if RL1 is initialized with alpha=0=rho, then it corresponds to unregularized ("vanilla") optimization

begin
# Training arguments are initialized
args = TrainArgs(; T=Float32) 

# Teacher and Student models are initialized and data is generated from the Gaussian distribution, whose mean value is computed by the teacher network.
train_set, validation_set, test_set, tstate, loss_fctn, args, teacher_tstate = setup_data_teacher_and_student(args; architecture_teacher=args.architecture_teacher, architecture_student=args.architecture_student, seed_teacher=35, seed_student=43, seed_train_set=1, seed_val_set=2, seed_test_set=2, teacher_weight_scaling=2, loss_fctn = Lux.MSELoss(), opt = Optimisers.Adam)

# Hyperparameter \alpha, which regulates the codelengthLoss-to-modelsize ratio in the loss function, is rescaled by the dataset size because we average the (codelength) loss over dataset size as well during training. \rho, which determines the L2 norm penalization, is rescaled in the same way.
scale_alpha_rho!(args, train_set, loss_fctn) 
end

# Plotting
plot_data_teacher_and_student(tstate,teacher_tstate, train_set) # if the domain and codomain of the networks are sufficiently low-dimensional (e.g. 2 input dimensions and 1 output dimension), then this plots two hypersurfaces, corresponding to the 2 functions that are computed by the networks. Furthermore, it also plots the dataset that was sampled from the teacher network.
plot_weights(teacher_tstate) # this plots the weights and biases of the teacher network. Colors indicate the sign of the connections and their thickness indicates their strength.
plot_weights(tstate)

# run one of the following procedures.
# One should not run them one after another without re-initializing the teacher-student networks (by re-running the functions above)
@time tstate, logs, loss_fun = RL1_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args);
@time tstate, logs, loss_fun = DRR_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args);
@time tstate, logs, loss_fun = PMMP_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args);
@time tstate, logs, loss_fun = layerwise_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args);

# Visalize Results
plot_data_teacher_and_student(tstate,teacher_tstate, train_set)
plot_weights(tstate)

PlotlyJS.plot(logs["epochs"], logs["train_loss"], label="train_loss")

p = PlotlyJS.plot(
    [
        PlotlyJS.scatter(x=logs["epochs"], y=logs["train_loss"], mode="lines", name="train_loss"),
        PlotlyJS.scatter(x=logs["epochs"], y=logs["val_loss"], mode="lines", name="val_loss")
    ],
    Layout(title="Training and Validation Loss", xaxis_title="Epochs", yaxis_title="Loss")
);
display(p);

PlotlyJS.plot(get1(1,logs["test_loss"]), get1(2,logs["test_loss"]), label="test_loss")
