"""
A simple script to run, test and develop the different optimization procedures in the Classifier Setup.
"""

using CUDA
include("../TrainArgs_classifier.jl")
include("HelperFunctions/loss_functions.jl")
include("HelperFunctions/generate_networks_and_data.jl")
include("../DatasetsModels/DatasetsModels.jl")


using .DatasetsModels: MNIST_data, CIFAR_data

# Load the different L0_regularization procedures
include("RL1_procedure.jl")
include("DRR_procedure.jl")
include("PMMP_procedure.jl")
include("FPP_procedure.jl")
# Note that if RL1 is initialized with alpha=0=rho, then it corresponds to unregularized ("vanilla") optimization

# Training arguments are initialized
args = TrainArgs(; T=Float32);

## Load one of the following datasets
train_set, validation_set, test_set = MNIST_data(args.train_batch_size, args.dev; seed=123)
# train_set, validation_set, test_set = CIFAR_data(args.train_batch_size, args.dev; train_set_size=args.train_set_size, val_set_size=args.val_set_size, test_set_size=args.test_set_size, seed=123);

args.val_batch_size = size(validation_set[1][2])[2]
args.test_batch_size = size(test_set[1][2])[2]
args.α = 1f-4
args.min_epochs = 100
model_seed = 42; loss_fctn = logitcrossentropy;
args.gradient_repetition_factor = 5

## Initialize one of the following models
# model = Lenet_5_Caffe()
# model = Lenet_MLP(Lux.sigmoid_fast; hidden_layer_sizes=[20, 20])
model = Lenet_MLP(Lux.sigmoid_fast)
# model = VGG(dropout=0.0f0);
# model = resnet()
# model = alexnet()

initial_parameter_count = Lux.parameterlength(model)

tstate = generate_tstate(model, model_seed, args.optimizer(args.lr); dev=args.dev);

# run one of the following procedures.
# One should not run them one after another without re-initializing the networks (by re-running the functions above)
@time tstate, logs, loss_fun = FPP_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args);
@time tstate, logs, loss_fun = RL1_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args);
@time tstate, logs, loss_fun = DRR_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args)
@time tstate, logs, loss_fun = PMMP_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args);

# Visalize Results
using PlotlyJS

p1 = PlotlyJS.plot(logs["epochs"], logs["train_loss"], label="train_loss")
p2 = PlotlyJS.plot(logs["epochs"], logs["val_loss"], label="val_loss")
p3 = PlotlyJS.plot(logs["epochs"], logs["validation_accuracy"], label="val_loss")
p4 = PlotlyJS.plot(logs["epochs"], logs["l0_mask"], label="l0_mask")

compression_rate =  initial_parameter_count / logs["l0_mask"][end]

q = PlotlyJS.plot(
    [
        PlotlyJS.scatter(x=logs["epochs"], y=logs["train_loss"], mode="lines", name="train_loss"),
        PlotlyJS.scatter(x=logs["epochs"], y=logs["val_loss"], mode="lines", name="val_loss"),
        PlotlyJS.scatter(x=logs["epochs"], y=logs["validation_accuracy"], mode="lines", name="validation_accuracy")
    ],
    Layout(title="without dropout but 50000 train_set", xaxis_title="Epochs", yaxis_title="Loss")
);
display(q);