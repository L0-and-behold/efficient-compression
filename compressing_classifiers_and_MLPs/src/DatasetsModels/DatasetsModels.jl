module DatasetsModels

using Base.Iterators: partition
using Flux
using Flux: gpu, Chain, Conv, Dense, relu, MaxPool, flatten, softmax, onehotbatch, onecold, logitcrossentropy
using CUDA
using Statistics: mean
using MLDatasets: CIFAR10

include("mnist_dataset.jl")
export get_dataset, MNIST_data, MNIST_trn_tst_switched, MNIST_custom_split, four_dim_array, hot_batch_encode

include("cifar_dataset.jl")
export CIFAR_data

include("imagenet_dataset.jl")
export imagenet_data

end # module