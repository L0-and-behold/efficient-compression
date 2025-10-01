using Lux
using Random
using Distributions: MvNormal
# using Base.Iterators: partition
# using OneHotArrays
using Optimisers
using LuxCUDA, Revise
using Boltz
using CUDA
using Metalhead

include("loss_functions.jl")

"""
    generate_dense_network(
        architecture::Vector{Int64};
        activations = nothing,
        dev = Lux.cpu_device(), 
        use_bias = Lux.True(),
        seed = 42,
        scaling = 1)

    Generates a densely connected multi-layer-perceptron (MLP) with a given architecture.

    Arguments:
        - `architecture`: The MLP architecture. It consists of an array of numbers that denote the layer dimensions. For example, one could choose [2,8,5,1]
        - `activations`: Vector of activation function. Default is `nothing`, which generates a network, where each activation function, except for the last one, is a tanh-activation function, while the last activation function is simply the identity.
        - `dev`: The device. Either `Lux.cpu_device()` or `Lux.gpu_device()`
        - `use_bias`: Whether biases are included in the MLP. Default is `Lux.True()`
        - `seed`: Seed that makes sure that the initial parameters are the same whenever this function is run with the same seed.
        - `scaling`: Scales weights in certain layers to make the function that the network computes more interesting. We use this in particular to make the teacher network function more non-linear. Default is `1` (no multiplicative scaling). 
"""
function generate_dense_network(
    architecture::Vector{Int64};
    activations = nothing,
    dev = Lux.cpu_device(), 
    use_bias = Lux.True(),
    seed = 42,
    scaling = 1)

    if isnothing(activations)
        activations = vcat([Lux.tanh_fast for _ in 3:length(architecture)]..., Lux.identity)
    end
    @assert length(architecture) - 1 == length(activations)
    
    layers = []
    for i in eachindex(activations)
        push!(layers, Lux.Dense(architecture[i] => architecture[i+1], activations[i]; use_bias=use_bias))
    end

    m = Lux.Chain(layers...; name="teacher-student network")

    rng = Random.default_rng()
    Random.seed!(rng, seed)
    ps, st = Lux.setup(rng, m) |> dev

    if scaling != 1
        # dtype = typeof(ps[1].bias[1])
        ps[1].weight .= ps[1].weight .* scaling .+ scaling/2
        ps[2].weight .= ps[2].weight .* scaling #.+ dtype(scaling)
    end

    return m, ps, st
end

"""
    generate_dataset(set_size::Int, 
        batch_size::Int,
        m::Union{Lux.Chain},
        ps::NamedTuple,
        st::NamedTuple;
        dtype::DataType = Float32,
        sigma::Number = dtype(1e-10),
        Σ = nothing,
        dev = Lux.cpu_device(),
        seed = 42)

    Generates a dataset from a given parameters-state configuration of a teacher network. It samples parameters from a conditional Gaussian distribution with standard deviation sigma or covariance matrix Σ and with mean value computed by the teacher network parameters.

    Arguments:
        - `set_size`: The desired dataset size.
        - `batch_size`: The desired batch size.
        - `m`: The neural network model that computes the mean value.
        - `ps`: The parameters of the neural model.
        - `st`: The states of the neural model.
        - `dtype`: The DataType (default is Float32)
        - `sigma`: The standard deviation. In that case, the covariance matrix is assumed to equal `Id .* sigma^2`
        - `Σ`: The covariance matrix, if a more complicated covariance structure is desired.
        - `dev`: The device. Either `Lux.cpu_device()` or `Lux.gpu_device()`
        - `seed`: Seed that makes sure that the initial parameters are the same whenever this function is run with the same seed.
"""
function generate_dataset(set_size::Int, 
    batch_size::Int,
    m::Union{Lux.Chain},
    ps::NamedTuple,
    st::NamedTuple;
    dtype::DataType = Float32,
    sigma::Number = dtype(1e-10),
    Σ = nothing,
    dev = Lux.cpu_device(),
    seed = 42)

    @assert isa(sigma, dtype)
    @assert set_size > 0
    @assert batch_size > 0
    @assert set_size % batch_size == 0 "set_size must be divisible by batch_size, got set_size = $set_size, batch_size = $batch_size"
    @assert sigma > 0 "sigma must be greater than 0, got sigma = $sigma"

    in_dims = m[1].in_dims
    out_dims = m[end].out_dims

    dataset = Tuple{Matrix{dtype}, Matrix{dtype}}[] |> dev
    number_batches = Int(set_size / batch_size)

    rng = MersenneTwister()
    Random.seed!(rng, seed)

    for _ in 1:number_batches
        x = 2 * rand(rng, dtype, in_dims, batch_size) .- 1 |> dev # uniformly sample 

        # we draw from a normal distribution with mean determined by teacher network and standard deviation noise with equal variance in all output dimensions. Alternatively, a covariance matrix can be passed as input to the data generating function.
        if isnothing(Σ)
            Σ = [i==j ? sigma^2 : zero(dtype) for i in 1:out_dims, j in 1:out_dims]
        else
            @assert isa(Σ, Matrix{dtype})
        end
        p = m(x, ps, st)[1] |> Lux.cpu_device()
        y = hcat([rand(rng, MvNormal(p[:,i], Σ), 1) for i in 1:batch_size]...) |> dev
        
        push!(dataset, (x, y))
    end
    return dataset |> dev
end

"""
    setup_data_teacher_and_student(args; architecture_teacher=[2,5,7,1], architecture_student=[2,25,25,1], seed_teacher=35, seed_student=43, seed_train_set=1, seed_val_set=2, seed_test_set=3, teacher_weight_scaling=2, loss_fctn = Lux.MSELoss(), opt = Optimisers.Adam)

    This function combines the functions generate_dense_network and generate_dataset to create a complete teacher-student setup.
    It first creates teacher and student networks of a desired architecture and thereafter samples data from the teacher, on which the student can be trained.
    
    Returns train_set, val_set, test_set, tstate, loss_fctn, args, teacher_tstate.
    The tstates contain all model and parameter information of the networks.
"""
function setup_data_teacher_and_student(args; architecture_teacher=args.architecture_teacher, architecture_student=args.architecture_student, seed_teacher=args.seed+35, seed_student=args.seed+43, seed_train_set=args.seed+1, seed_val_set=args.seed+2, seed_test_set=args.seed+3, teacher_weight_scaling=2, loss_fctn = Lux.MSELoss(), opt = Optimisers.Adam)
  
    teacher_m, teacher_ps, teacher_st = generate_dense_network(architecture_teacher; seed=seed_teacher, scaling=teacher_weight_scaling, dev=args.dev)
    m, ps, st = generate_dense_network(architecture_student; seed=seed_student, dev=args.dev)

    teacher_tstate = Training.TrainState(teacher_m, teacher_ps, teacher_st, opt(args.lr))
    tstate = Training.TrainState(m, ps, st, opt(args.lr))

    train_set = generate_dataset(Int64(args.train_set_size), 
        Int64(args.train_batch_size), 
        teacher_m, 
        teacher_ps, 
        teacher_st; 
        dtype = args.dtype,
        sigma = args.noise,
        dev = args.dev,
        seed = seed_train_set)
    val_set = generate_dataset(args.val_set_size, 
        args.val_batch_size,
        teacher_m, 
        teacher_ps, 
        teacher_st;
        dtype = args.dtype,
        sigma = args.noise,
        dev = args.dev,
        seed = seed_val_set)
    test_set = generate_dataset(args.test_set_size, 
        args.test_batch_size,
        teacher_m, 
        teacher_ps, 
        teacher_st;
        dtype = args.dtype,
        sigma = args.noise,
        dev = args.dev,
        seed = seed_test_set)
    
    return train_set, val_set, test_set, tstate, loss_fctn, args, teacher_tstate
end

function scale_alpha_rho!(args, train_set, loss_fctn)
    num_batches = args.dtype(length(train_set))
    num_elements_in_batch = args.dtype(size(train_set[1][1])[end])
    args.α /= num_batches
    args.L1_alpha /= num_batches
    args.ρ /= num_batches
    if loss_fctn == Lux.MSELoss() || loss_fctn == logitcrossentropy # those aggregate data with mean
        args.α /= num_elements_in_batch
        args.L1_alpha /= num_elements_in_batch
        args.ρ /= num_elements_in_batch
    end
end

function generate_tstate(model, seed, opt; dev=Lux.gpu_device())
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    ps, st = Lux.setup(rng, model) |> dev
    tstate = Lux.Training.TrainState(model, ps, st, opt)
    return tstate
end


#### Below are image classifier models

### For MNIST

"""
    Lenet_MLP(activation = Lux.relu; hidden_layer_sizes = [300, 100])

    Generates a Lenet-300-100, fully connected

    following https://github.com/Germoe/LeNet-300-100/blob/main/LeNet-300-100-LTH-Paper.ipynb
    # and https://github.com/panweihit/DropNeuron/blob/master/lenet-300-100.py (but without dropout)
"""
function Lenet_MLP(activation = Lux.relu; hidden_layer_sizes = [300, 100])
    layers = []
    push!(layers, Lux.FlattenLayer())
    push!(layers, Lux.Dense(784, hidden_layer_sizes[1], activation))
    push!(layers, Lux.Dense(hidden_layer_sizes[1], hidden_layer_sizes[2], activation))
    push!(layers, Lux.Dense(hidden_layer_sizes[2], 10, identity))
    return Lux.Chain(layers; name="Lenet-" * string(hidden_layer_sizes[1]) * "-" * string(hidden_layer_sizes[2]))
end


"""
    Lenet_5(input_image_resolution = 32, activation=Lux.sigmoid_fast)

    Generates Lenet-5 architecture according to Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.
    See also: https://penkovsky.com/neural-networks/day5/
"""
function Lenet_5(input_image_resolution = 32, activation=Lux.sigmoid_fast)
    layers = []
    conv_out_dim = Int(((input_image_resolution-4)/2-4)/2)
    push!(layers, Lux.Conv((5, 5), (1 => 6), activation; stride=1, pad=0, dilation=1, groups=1, use_bias=Lux.True(), cross_correlation=Lux.True()))
    push!(layers, Lux.MaxPool((2,2); stride=(2,2), pad=0, dilation=1))
    push!(layers, Lux.Conv((5, 5), (6 => 16), activation; stride=1, pad=0, dilation=1, groups=1, use_bias=Lux.True(), cross_correlation=Lux.True()))
    push!(layers, Lux.MaxPool((2,2); stride=(2,2), pad=0, dilation=1))
    push!(layers, Lux.FlattenLayer())
    push!(layers, Lux.Dense(conv_out_dim^2*16,120, activation))
    push!(layers, Lux.Dense(120,84, activation))
    push!(layers, Lux.Dense(84,10, identity))
    Lenet5model = Lux.Chain(layers; name="Lenet-5")
    return Lenet5model
end

"""
    Lenet_5_Caffe(activation=Lux.relu; input_image_resolution=28) 

    Generates Lenet-5-Caffe architecture according to https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet_train_test.prototxt
    See also: https://github.com/AMLab-Amsterdam/L0_regularization/blob/master/models.py (but without beta_ema)
"""
function Lenet_5_Caffe(activation=Lux.relu; input_image_resolution=28) 
    layers = []
    conv_out_dim = Int(((input_image_resolution-4)/2-4)/2)
    push!(layers, Lux.Conv((5, 5), (1 => 20), activation; stride=1, pad=0, dilation=1, groups=1, use_bias=Lux.True(), cross_correlation=Lux.True())) # we follow the pytorch implementation where cross_correlation=True()
    push!(layers, Lux.MaxPool((2,2); stride=(2,2), pad=0, dilation=1))
    push!(layers, Lux.Conv((5, 5), (20 => 50), activation; stride=1, pad=0, dilation=1, groups=1, use_bias=Lux.True(), cross_correlation=Lux.True()))
    push!(layers, Lux.MaxPool((2,2); stride=(2,2), pad=0, dilation=1))
    push!(layers, Lux.FlattenLayer())
    push!(layers, Lux.Dense(conv_out_dim^2*50,500, activation))
    push!(layers, Lux.Dense(500,10, identity))
    return Lux.Chain(layers; name="Lenet-5-Caffe")
end


### For CIFAR-10 and Imagenet


"""
    VGG(input_dims = (32,32), in_channels = 3; depth=16, nclasses=10, batchnorm=true, fcsize=512, dropout=0.4f0)

    Generates VGG architecture following https://www.kaggle.com/code/vtu5118/cifar-10-using-vgg16 (but with both fc layers of equal size like in the original architecture https://arxiv.org/abs/1409.1556)
"""
function VGG(input_dims = (32,32), in_channels = 3; depth=16, nclasses=10, batchnorm=true, fcsize=512, dropout=0.4f0)
    return Vision.VGG(input_dims; config=Vision.VGG_CONFIG[depth], inchannels=in_channels, batchnorm=batchnorm, nclasses=nclasses, fcsize=fcsize, dropout=dropout)
end

function resnet(; depth=50) # for Imagenet
    return Vision.ResNet(depth; pretrained=false)
end

function alexnet() # for Imagenet
    return Vision.AlexNet(; pretrained=false)
end

## WideResNet
# # Define a basic wide residual block
# function wide_basic_block(in_channels, out_channels, stride, widening_factor)
#     mid_channels = out_channels * widening_factor
#     layers = Lux.Chain(
#         Lux.Conv((3, 3), (in_channels => mid_channels), Lux.identity; stride=stride, pad=1, use_bias=Lux.False()),
#         Lux.BatchNorm(mid_channels),
#         Lux.relu,
#         Lux.Conv((3, 3), (mid_channels => out_channels), Lux.identity; stride=1, pad=1, use_bias=Lux.False()),
#         Lux.BatchNorm(out_channels)
#     )
#     if stride != 1 || in_channels != out_channels
#         downsample = Lux.Conv((1, 1), (in_channels => out_channels), Lux.identity; stride=stride, use_bias=Lux.False())
#         return Lux.Chain(layers, downsample)
#     else
#         return layers
#     end
# end

# # Define the Wide ResNet architecture
# function WideResNet(num_classes, depth, widening_factor)
#     layers = []
#     # Initial Conv Layer
#     push!(layers, Lux.Conv((3, 3), (3 => 16), Lux.identity; stride=1, pad=1, use_bias=Lux.False()))
#     push!(layers, Lux.BatchNorm(16))
#     push!(layers, Lux.relu)

#     num_blocks_per_stage = (depth - 4) ÷ 6
#     stages = [16, 32, 64]

#     for (i, out_channels) in enumerate(stages)
#         stride = (i == 1) ? 1 : 2
#         for _ in 1:num_blocks_per_stage
#             push!(layers, wide_basic_block((i == 1) ? 16 : stages[i - 1] * widening_factor, out_channels, stride, widening_factor))
#             stride = 1  # Only the first block in each stage uses the stride
#         end
#     end

#     push!(layers, Lux.GlobalMeanPool())
#     push!(layers, Lux.FlattenLayer())
#     push!(layers, Lux.Dense(64 * widening_factor, num_classes))

#     return Lux.Chain(layers...)
# end

# # Define the model
# num_classes = 10
# depth = 28
# widening_factor = 10
# modelwrn = WideResNet(num_classes, depth, widening_factor)

# # Display model summary
# @info model
