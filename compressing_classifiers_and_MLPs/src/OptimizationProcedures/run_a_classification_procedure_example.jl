"""
A simple script to run, test and develop the different optimization procedures in the Classifier Setup.
"""

using CUDA
include("../TrainArgs_classifier.jl")
include("HelperFunctions/loss_functions.jl")
include("HelperFunctions/generate_networks_and_data.jl")
include("../DatasetsModels/DatasetsModels.jl")

using ParameterSchedulers: Step

using .DatasetsModels: MNIST_data, CIFAR_data, imagenet_data

# Load the different L0_regularization procedures
include("RL1_procedure.jl")
include("DRR_procedure.jl")
include("PMMP_procedure.jl")
include("FPP_procedure.jl")
# Note that if RL1 is initialized with alpha=0=rho, then it corresponds to unregularized ("vanilla") optimization

# Training arguments are initialized
args = TrainArgs(; T=Float32);

## Load one of the following datasets
# train_set, validation_set, test_set = MNIST_data(args.train_batch_size, args.dev; seed=123)
# train_set, validation_set, test_set = CIFAR_data(args.train_batch_size, args.dev; seed=123);

## IMAGENET
include("../DatasetsModels/imagenet/imagenet_path.jl")
args.train_batch_size = 256
args.val_batch_size = 256
args.test_batch_size = 256
image_size = 224
train_set, validation_set, test_set = imagenet_data(imagenet_path, args.train_batch_size, args.val_batch_size, image_size; dev=gpu_device())
args.train_set_size = length(train_set) * args.train_batch_size
args.val_set_size = 50000
args.test_set_size = 50000
args.lr = 0.1f0
args.min_epochs = 100
args.max_epochs = 100
args.optimizer = lr -> Momentum(lr, 0.9f0)
args.schedule = Step(
    args.lr,                                   # Initial learning rate
    0.1f0,                                     # Decay factor (multiply by 0.1 = divide by 10)
    30                   # [30, 60, 90]        # Epochs where decay happens
)
## IMAGENET END

# args.val_batch_size = size(validation_set[1][2])[2]
# args.test_batch_size = size(test_set[1][2])[2]
args.α = 1f-4
model_seed = 42; loss_fctn = logitcrossentropy;
args.gradient_repetition_factor = 5

## Initialize one of the following models
# model = Lenet_5_Caffe()
# model = Lenet_MLP(Lux.sigmoid_fast; hidden_layer_sizes=[20, 20])
# model = Lenet_MLP(Lux.sigmoid_fast)
# model = VGG(dropout=0.0f0);
model = resnet()
# model = alexnet()

initial_parameter_count = Lux.parameterlength(model)


tstate = generate_tstate(model, model_seed, args.optimizer(args.lr); dev=args.dev);

# run one of the following procedures.
# One should not run them one after another without re-initializing the networks (by re-running the functions above)
@time tstate, logs, loss_fun = FPP_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args);
@time tstate, logs, loss_fun = RL1_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args);
@time tstate, logs, loss_fun = DRR_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args)
@run tstate, logs, loss_fun = PMMP_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args);

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

## check

# trainloss_converged_at = logs["converged_at"][1]
# turningPointsAfterConvergence = logs["turning_points_val_loss"][logs["turning_points_val_loss"].>trainloss_converged_at]
# best_val_loss_during_finetuning = minimum(logs["val_loss"][turningPointsAfterConvergence.-1])

# function loss_on_dataset(dataset)::Number
#     vjp = AutoZygote()
#     total_loss = zero(args.dtype)
#     for batch in dataset
#         _, loss, _, _ = Lux.Training.compute_gradients(vjp, loss_fun, batch, tstate)
#         total_loss += loss
#     end
#     return total_loss / args.dtype(length(dataset))
# end

# final_val_loss_of_returned_model = loss_on_dataset(validation_set)
# function recursive_sum(mask, l0_mask)
#     for m in mask
#         if isa(m, NamedTuple) && !isempty(m)
#             l0_mask = recursive_sum(m, l0_mask)
#         elseif isa(m, AbstractArray)
#             l0_mask += sum(m)
#         end
#     end
#     return l0_mask
# end
# l0_mask= recursive_sum(tstate.states.mask, args.dtype(0))
# Lux.parameterlength(tstate.model) / l0_mask
