
using Lux: gpu_device, cpu_device

"""
    CIFAR_data(batch_size::Int, dev=gpu_device(); train_set_size=45000, val_set_size=5000, test_set_size=10000, seed=1234)

    Returns the CIFAR 10 data set in a 45k/5k/10k split, processed, split into batches and ready for training on GPU.

    Device parameter is a function that moves the data to the desired device (cpu or gpu) e.g. Flux.gpu or Flux.cpu or Lux.cpu_device() or Lux.gpu_device()
"""
function CIFAR_data(batch_size::Int, dev=gpu_device(); train_set_size=45000, val_set_size=5000, test_set_size=10000, seed=1234)

    train_and_val_imgs, train_and_val_labels = CIFAR10(split=:train)[:]
    test_imgs, test_labels = CIFAR10(split=:test)[:]   
    
    @assert train_set_size + val_set_size <= length(train_and_val_labels)
    @assert test_set_size <= length(test_labels)

    Random.seed!(seed)
    train_and_val_indices = randperm(length(train_and_val_labels))
    train_indices = train_and_val_indices[1:train_set_size]
    val_indices = train_and_val_indices[train_set_size+1:train_set_size+val_set_size]
    test_indices = randperm(length(test_labels))[1:test_set_size]

    # Train, Validiation, Test split
    train_imgs = train_and_val_imgs[:, :, :, train_indices]
    train_labels = train_and_val_labels[train_indices]
    val_imgs = train_and_val_imgs[:, :, :, val_indices]
    val_labels = train_and_val_labels[val_indices]
    test_imgs = test_imgs[:, :, :, test_indices]
    test_labels = test_labels[test_indices]

    # Mini Batches
    mb_idxs = partition(1:length(train_labels), batch_size)
    train_set = ([(train_imgs[:,:,:,i], train_labels[i]) for i in mb_idxs])
    val_set = ([(val_imgs[:,:,:,:], val_labels[:])])
    mb_test_idxs = partition(1:length(test_labels), Int(length(test_labels)/2))
    test_set = ([(test_imgs[:,:,:,i], test_labels[i]) for i in mb_test_idxs])

    # Encode labels and move data to device
    function _hot_batch_encode(data)
        new_data = []
        for d in data
            encoded_data = (d[1] , onehotbatch(d[2], 0:9))
            push!(new_data, encoded_data)
        end
        return new_data
    end

    train_set = _hot_batch_encode(train_set)
    val_set = _hot_batch_encode(val_set)
    test_set = _hot_batch_encode(test_set)

    train_set = train_set |> dev
    val_set = val_set |> dev
    test_set = test_set |> dev

    return train_set, val_set, test_set
end

