using Random
import Flux

function get_model(args)
    if args.architecture == "CNNmodel"
        println("using CNNmodel")
        return build_CNNmodel(args)
    elseif args.architecture == "LeNet300100"
        println("using LeNet300100")
        return LeNet300100(args)
    elseif args.architecture == "ciresan2010deep"
        println("using ciraccuracy_is_saturatedesan2010deep_model")
        return ciresan2010deep_model(args)
    else
        error("unknown model: $(args.model)")
    end
end


"""
    build_CNNmodel(args; imgsize = (28,28,1), nclasses = 10, device = gpu, seed=rand(Int32))

    Returns the model first used for initial testing of the method. It does not appear in any external reference.
"""
function build_CNNmodel(args; imgsize = (28,28,1), nclasses = 10, device = gpu, seed=rand(Int32))
    rng = Random.MersenneTwister(seed)
    init = Flux.glorot_uniform(rng)
    init(dims...) = Flux.glorot_uniform(rng, dims...)

    return Chain(
        Conv((5, 5), 1=>50, relu, init=init),  # conv1
        MaxPool((2, 2)),  # pool1
        Conv((5, 5), 50=>100, relu, init=init),  # conv2
        MaxPool((2, 2)),  # pool2
        flatten,
        Dense(100*4*4, 500, init=init),  # ip1
        relu,  # relu1
        Dense(500, nclasses, init=init),  # ip2
    )  |> device
end


"""
    LeNet300100(; imgsize = (28,28,1), nclasses = 10, device = gpu, seed=rand(Int32))

    Model used in Oliveira et. al. paper on MNIST dataset
"""
function LeNet300100(; imgsize = (28,28,1), nclasses = 10, device = gpu, seed=rand(Int32))
    rng = Random.MersenneTwister(seed)
    init = Flux.glorot_uniform(rng)
    init(dims...) = Flux.glorot_uniform(rng, dims...)

    return Chain(
        flatten,
        Dense(784, 300, init=init),
        relu,
        Dense(300, 100, init=init),
        relu,
        Dense(100, 10, init=init),
    ) |> device
end

function LeNet300100(args; imgsize = (28,28,1), nclasses = 10, device = gpu, seed=rand(Int32))
    return LeNet300100(imgsize = imgsize, nclasses = nclasses, device = device, seed=seed)
end

scaled_tanh(x) = (A = 1.7159f0; B = 0.6666f0; A .* tanh.(B .* x))

"""
    ciresan2010deep_model(args; imgsize = (28,28,1), nclasses = 10, device = gpu, seed=rand(Int32))

    Returns the 2nd architecture from Ciresan et. al. 2010 paper "Deep, Big, Simple neural Nets for Handwritten Digit Recognition"
"""
function ciresan2010deep_model(args; imgsize = (28,28,1), nclasses = 10, device = gpu, seed=rand(Int32))
    rng = Random.MersenneTwister(seed)
    init = Flux.glorot_uniform(rng)
    init(dims...) = Flux.glorot_uniform(rng, dims...)
    
    return Chain(
        flatten,
        Dense(784, 1500, init=init),
        scaled_tanh,
        Dense(1500, 1000, init=init),
        scaled_tanh,
        Dense(1000, 500, init=init),
        scaled_tanh,
        Dense(500, nclasses, init=init),
    ) |> device
end


"""
    dense_3x512(args; imgsize = (32,32,3), nclasses = 10)

    Returns a dense ReLu model with 3 hidden layers
"""
function dense_3x512(args; imgsize = (32,32,3), nclasses = 10)
    device = args.dev
    return dense_3x512(; imgsize = imgsize, nclasses = nclasses, device = device)
end

"""
    dense_3x512(; imgsize = (32,32,3), nclasses = 10, device = gpu, seed=rand(Int32))

    Returns a dense ReLu model with 3 hidden layers
"""
function dense_3x512(; imgsize = (32,32,3), nclasses = 10, device = gpu, seed=rand(Int32))
    rng = Random.MersenneTwister(seed)
    init = Flux.glorot_uniform(rng)
    init(dims...) = Flux.glorot_uniform(rng, dims...)
    
    return Chain(
        flatten,
        Dense(prod(imgsize), 512, init=init),
        relu,
        Dense(512, 512, init=init),
        relu,
        Dense(512, 512, init=init),
        relu,
        Dense(512, nclasses, init=init),
    ) |> device
end