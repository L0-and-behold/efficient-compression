module DatasetsModels

    using Lux, Random, Distributed, OneHotArrays, CUDA, DataAugmentation, FileIO
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

    include("imagenet_dataset.jl")
    export imagenet_data

    include("toy_imagenet_dataset.jl")
    export toy_imagenet_data

end # module