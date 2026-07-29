"""
    CIFAR_C_data(corruption_type::String, path_to_CIFAR_C="./src/DatasetsModels/CIFAR-C/CIFAR-10-C/", dev=gpu_device())

    Returns the CIFAR-C 10 test data set in a 50k split, processed, split into batches

    CIFAR-C can be used to test robustness to corruptions and pertubations, cf. https://github.com/hendrycks/robustness

    To use this function, you need to download CIFAR-10-C.tar from https://zenodo.org/records/2535967, then unpack it and move all .npy files to "./CIFAR-C/CIFAR-10-C/" (where ./ denotes the DatasetsModels directory, in which this `cifar-c_dataset.jl` file is)
    
    `corruption_type` denotes the type of corruption or perturbation applied to the original cifar image. It is one of the following 15 types
        - `brightness`, 
        - `contrast`, 
        - `defocus_blur`, 
        - `elastic_transform`, 
        - `fog`, 
        - `frost`,  
        - `gaussian_noise`,
        - `glass_blur`, 
        - `impulse_noise`,
        - `jpeg_compression` 
        - `motion_blur`, 
        - `pixelate`, 
        - `shot_noise`, 
        - `snow`, 
        - `zoom_blur`, 
    Note that the CIFAR-C folder also contains 4 extra ("held-out") corruptions as well, namely `speckle_noise`, `gaussian_blur`, `spatter`, and `saturate`. These were added later as a way to test generalization to corruptions not used for tuning/validation. Including them would deviate from the standard benchmark.

    `path_to_CIFAR_C`: has to coincide with the relative path into which the CIFAR-C .npy files were unpacked

    `dev`: Device parameter is a function that moves the data to the desired device (cpu or gpu) e.g. Flux.gpu or Flux.cpu or Lux.cpu_device() or Lux.gpu_device()
"""

using NPZ

const corruption_types = [
    "brightness", 
    "contrast", 
    "defocus_blur", 
    "elastic_transform", 
    "fog", 
    "frost",  
    "gaussian_noise",
    "glass_blur", 
    "impulse_noise",
    "jpeg_compression", 
    "motion_blur", 
    "pixelate", 
    "shot_noise", 
    "snow", 
    "zoom_blur", 
]

function CIFAR_C_data(corruption_type::String, path_to_CIFAR_C="./src/DatasetsModels/CIFAR-C/CIFAR-10-C/", dev=gpu_device()) # ; test_set_size=50000, seed=1234)

    # Load data
    path_to_data = path_to_CIFAR_C * corruption_type * ".npy"
    path_to_labels = path_to_CIFAR_C * "labels.npy"
    test_imgs = npzread(path_to_data)  # shape (50000, 32, 32, 3), 5 severity levels × 10000 stacked
    test_labels = npzread(path_to_labels)

    # Bring data into the same format that one obtains when loading MLDatasets: CIFAR10
    test_imgs = Float32.(test_imgs) ./ 255f0
    test_imgs = permutedims(test_imgs, (3, 2, 4, 1)) # (32,32,3,50000) N,H,W,C -> W,H,C,N (Lux and Pytorch differ with respect to array processing format)
    test_labels = Int64.(test_labels)

    ## process data in the same way in which we processed MLDatasets: CIFAR10 data

    # # Random subset selection (we omit this for CIFAR-C because we always use the full dataset)
    # @assert test_set_size <= length(test_labels)
    # Random.seed!(seed)
    # test_indices = randperm(length(test_labels))[1:test_set_size]
    # test_imgs = test_imgs[:, :, :, test_indices]
    # test_labels = test_labels[test_indices]

    # Mini Batches
    mb_test_idxs = partition(1:length(test_labels), Int(length(test_labels)/5)) # there are 5 severity levels. Hence the batch size is set such that the partition cleanly separates the severity levels
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

    test_set = _hot_batch_encode(test_set)
    
    test_set = test_set |> dev

    return test_set
end

