module DatasetsModels

    using Lux, Random, Distributed, OneHotArrays, CUDA, DataAugmentation, FileIO, ImageIO, Images, JLD2, ProgressMeter, Printf
    
    using Flux: onehotbatch, gpu
    using Lux: DistributedUtils, gpu_device, cpu_device
    using Base.Iterators: partition
    using Statistics: mean
    using MLDatasets: CIFAR10, MNIST
    using MLUtils: DataLoader
    import Lux.DistributedUtils.DistributedDataContainer

    include("mnist_dataset.jl")
    export get_dataset, MNIST_data, MNIST_trn_tst_switched, MNIST_custom_split, four_dim_array, hot_batch_encode

    include("cifar_dataset.jl")
    export CIFAR_data

    # load the ImageNet submodule and re-export its API
    include("ImageNet/ImageNet.jl")
    using .ImageNet

    export imagenet_online_data, imagenet_chunked_data, preprocess_split, ChunkedImageNet, ChunkedBatch, DeviceDataLoader
    export toy_imagenet_data


end # module