"""
A simple script to run, test and develop the different optimization procedures in the Classifier Setup.
"""

using CUDA
includet("../TrainArgs_Lux.jl")
includet("HelperFunctions/loss_functions.jl")
includet("HelperFunctions/generate_networks_and_data.jl")
include("../DatasetsModels/DatasetsModels.jl")

using .DatasetsModels: MNIST_data, CIFAR_data

# Load the different L0_regularization procedures
includet("RL1_procedure.jl")
includet("DRR_procedure.jl")
includet("PMMP_procedure.jl")
# Note that if RL1 is initialized with alpha=0=rho, then it corresponds to unregularized ("vanilla") optimization

# Training arguments are initialized
args = TrainArgs(; T=Float32);

## Load one of the following datasets
# train_set, validation_set, test_set = MNIST_data(args.train_batch_size, args.dev; train_set_size=args.train_set_size, val_set_size=args.val_set_size, test_set_size=args.test_set_size, seed=123)
train_set, validation_set, test_set = CIFAR_data(args.train_batch_size, args.dev; train_set_size=args.train_set_size, val_set_size=args.val_set_size, test_set_size=args.test_set_size, seed=123);

model_seed = 42; loss_fctn = logitcrossentropy;

## Initialize one of the following models
# model = Lenet_5_Caffe()
# model = Lenet_MLP(Lux.sigmoid_fast; hidden_layer_sizes=[20, 20])
model = VGG(dropout=0.0f0);
Lux.parameterlength(model)

tstate = generate_tstate(model, model_seed, args.optimizer(args.lr); dev=args.dev);

# run one of the following procedures.
# One should not run them one after another without re-initializing the teacher-student networks (by re-running the functions above)
@time tstate, logs, loss_fun = RL1_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args);
@time tstate, logs, loss_fun = DRR_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args)
@time tstate, logs, loss_fun = PMMP_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args);

# Visalize Results
using PlotlyJS

p1 = PlotlyJS.plot(logs["epochs"], logs["train_loss"], label="train_loss")
p2 = PlotlyJS.plot(logs["epochs"], logs["val_loss"], label="val_loss")
p3 = PlotlyJS.plot(logs["epochs"], logs["validation_accuracy"], label="val_loss")

q = PlotlyJS.plot(
    [
        PlotlyJS.scatter(x=logs["epochs"], y=logs["train_loss"], mode="lines", name="train_loss"),
        PlotlyJS.scatter(x=logs["epochs"], y=logs["val_loss"], mode="lines", name="val_loss"),
        PlotlyJS.scatter(x=logs["epochs"], y=logs["validation_accuracy"], mode="lines", name="validation_accuracy")
    ],
    Layout(title="without dropout but 50000 train_set", xaxis_title="Epochs", yaxis_title="Loss")
);
display(q);
