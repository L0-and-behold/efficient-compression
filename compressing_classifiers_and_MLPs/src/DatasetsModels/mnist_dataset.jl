
using Flux: onehotbatch, gpu
using MLDatasets: MNIST
using Base.Iterators: partition
using Random

using Lux: gpu_device, cpu_device


"""
    get_dataset(args)

    Arguments:
    - args : A TrainArgs object containing the dataset name and the batch size.

    Returns:
    - dataset according to the logic defined by the _DATASET_FUNCTIONS dictionary.
"""
function get_dataset(args)

    _DATASET_FUNCTIONS = Dict(
        "MNIST" => MNIST_data,
        "MNIST_trn_tst_switched" => MNIST_trn_tst_switched,
        "MNIST_custom_split" => MNIST_custom_split
    )

    # in case of a custom split, the dataset name is a Tuple ("MNIST_custom_split", train_set_size)
    dataset_name = isa(args.dataset, Tuple) ? args.dataset[1] : args.dataset

    if haskey(_DATASET_FUNCTIONS, dataset_name)
        return _DATASET_FUNCTIONS[dataset_name](args)
    else
        println(typeof(args.dataset))
        error("unknown dataset: $(args.dataset)")
    end
end

"""
    MNIST_data(batch_size, dev=gpu_device(); train_set_size=45000, val_set_size=5000, test_set_size=10000, seed=123)

    Returns MNIST dataset, split into train, validation and test set and transfers them to the correct device.
"""
function MNIST_data(batch_size, dev=gpu_device(); train_set_size=45000, val_set_size=5000, test_set_size=10000, seed=123)

    train_and_val_imgs, train_and_val_labels = MNIST(split=:train)[:]
    test_imgs, test_labels = MNIST(split=:test)[:]       
    
    @assert train_set_size + val_set_size <= length(train_and_val_labels)
    @assert test_set_size <= length(test_labels)

    # Extract subsets
    Random.seed!(seed)
    train_and_val_indices = randperm(length(train_and_val_labels))
    train_indices = train_and_val_indices[1:train_set_size]
    val_indices = train_and_val_indices[train_set_size+1:train_set_size+val_set_size]
    test_indices = randperm(length(test_labels))[1:test_set_size]

    train_imgs = train_and_val_imgs[:, :, train_indices]
    train_labels = train_and_val_labels[train_indices]
    val_imgs = train_and_val_imgs[:, :, val_indices]
    val_labels = train_and_val_labels[val_indices]
    test_imgs = test_imgs[:, :, test_indices]
    test_labels = test_labels[test_indices]
    
    # Reshape the data to be 4D arrays
    train_imgs = four_dim_array(train_imgs)
    val_imgs = four_dim_array(val_imgs)
    test_imgs = four_dim_array(test_imgs)

    # Mini Batches
    mb_idxs = partition(1:length(train_labels), batch_size)
    train_set = ([(train_imgs[:,:,:,i], train_labels[i]) for i in mb_idxs])
    val_set = ([(val_imgs[:,:,:,:], val_labels[:])])
    test_set = ([(test_imgs[:,:,:,:], test_labels[:])])

    train_set = hot_batch_encode(train_set)
    val_set = hot_batch_encode(val_set)
    test_set = hot_batch_encode(test_set)

    train_set = train_set |> dev
    val_set = val_set |> dev
    test_set = test_set |> dev

    return train_set, val_set, test_set
end


"""
    MNIST_trn_tst_switched(args)

    Returns MNIST with training set of 10k pictures, test set of 60k: used to evaluate role of overfitting when expressing L = - ∑_x∈D log₂p(x) + l(p) for D the MNIST dataset and p a trained model
"""
function MNIST_trn_tst_switched(args)
    # role of training and test sets switched
    # @time begin
        test_imgs, test_labels = MNIST(split=:train)[:]
        train_imgs, train_labels = MNIST(split=:test)[:]       
    # end
    
    # @time begin
    # Reshape the data to be 4D arrays
    train_imgs = four_dim_array(train_imgs)
    test_imgs = four_dim_array(test_imgs)

    # Mini Batches
    bs = args.batch_size
    mb_idxs = partition(1:length(train_labels), bs)
    train_set = ([(train_imgs[:,:,:,i], train_labels[i]) for i in mb_idxs])
    test_set = ([(test_imgs[:,:,:,:], test_labels[:])])

    train_set = hot_batch_encode(train_set)
    test_set = hot_batch_encode(test_set)

    train_set = train_set |> gpu
    test_set = test_set |> gpu
    # end
    
    return train_set, test_set
end


"""
    MNIST_custom_split(args)

    Arguments:
    - args
        - args.dataset : A Tuple{String, Number} where the String should be "MNIST_custom_split" and the Number the size of the training set.

    Returns:
    - train_set : A vector of minibatches, where each minibatch is a tuple of a 4D-Array (images) and a 1D-Array (labels). Both are hot encoded.
    - test_set : A vector of minibatches, where each minibatch is a tuple of a 4D-Array (images) and a 1D-Array (labels). Both are hot encoded.
"""
function MNIST_custom_split(args)

    if !isa(args.dataset, Tuple{String, Number})
        error("MNIST_custom_split expects a Tuple{String, Number} as args.dataset")
    end
    train_set_size = args.dataset[2]
    batch_size = args.batch_size

    try 
        args.train_set_size = train_set_size
    catch
    end

    return MNIST_custom_split(train_set_size, batch_size)
end


"""
    MNIST_custom_split(train_set_size::Int, batch_size::Int)

    Arguments:
    - train_set_size::Int : Number of samples in the training set. Must be between 0 and 70000.

    Returns:
    - train_set : A vector of minibatches, where each minibatch is a tuple of a 4D-Array (images) and a 1D-Array (labels). Both are hot encoded.
    - test_set : A vector of minibatches, where each minibatch is a tuple of a 4D-Array (images) and a 1D-Array (labels). Both are hot encoded.
"""
function MNIST_custom_split(train_set_size::Int, batch_size::Int)
    if train_set_size >= 70000 || train_set_size <= 0
        error("train_set_size must be between 1 and 69999. Got $train_set_size")
    end
    if batch_size > train_set_size
        @warn "Batch size must be between 1 and train_set_size $train_set_size. Got $batch_size. Setting batch_size ot train_set_size..."
        batch_size = train_set_size
    elseif batch_size < 1
        error("Batch size must be between 1 and train_set_size $train_set_size. Got $batch_size")
    end

    train_imgs, train_labels = MNIST(split=:train)[:]
    test_imgs, test_labels = MNIST(split=:test)[:]
    
    # Redistribute the data according to the train_set_size
    all_imgs = cat(train_imgs, test_imgs, dims=3)
    all_lables = cat(train_labels, test_labels, dims=1) 
    train_imgs, test_imgs = all_imgs[:, :, 1:train_set_size], all_imgs[:, :, train_set_size+1:end]
    train_labels, test_labels = all_lables[1:train_set_size], all_lables[train_set_size+1:end]
    
    # Reshape the images to be 4D arrays
    train_imgs = four_dim_array(train_imgs)
    test_imgs = four_dim_array(test_imgs)

    # Mini Batches
    mb_idxs = partition(1:length(train_labels), batch_size)
    train_set = ([(train_imgs[:,:,:,i], train_labels[i]) for i in mb_idxs])
    test_set = ([(test_imgs[:,:,:,:], test_labels[:])])

    train_set = hot_batch_encode(train_set)
    test_set = hot_batch_encode(test_set)

    train_set = train_set |> gpu
    test_set = test_set |> gpu

    return train_set, test_set
end

### Helper functions

function four_dim_array(x::Array{<:Any, 3}) 
    return reshape(x, size(x, 1), size(x, 2), 1, size(x, 3))
end

function hot_batch_encode(data)
    new_data = []
    for i in 1:length(data)
        push!( new_data, (data[i][1] , onehotbatch(data[i][2], 0:9)) )
    end
    return new_data
end