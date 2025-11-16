using MLUtils: DataLoader
using OneHotArrays
using Random
using Lux

"""
    SimpleToyDataset(n_samples, n_classes, image_dims; rng=Random.GLOBAL_RNG)

A minimal in‑memory dataset that yields random images of shape `image_dims`
(H×W×C) and one‑hot class labels. It is purpose‑built for quick debugging –
no file I/O, no complex augmentation pipeline.
"""
struct SimpleToyDataset{R<:AbstractRNG}
    n_samples::Int
    n_classes::Int
    image_dims::NTuple{3,Int}   # (height, width, channels)
    rng::R
end

Base.length(ds::SimpleToyDataset) = ds.n_samples

function Base.getindex(ds::SimpleToyDataset, i::Int)
    # deterministic seed per index for reproducibility
    rng = MersenneTwister(hash(i) + hash(ds.rng))
    img = rand(rng, Float32, ds.image_dims...)
    label = (i - 1) % ds.n_classes
    return img, OneHotArrays.onehot(label, 0:(ds.n_classes - 1))
end

"""
    construct_toy_dataloaders(n_train, n_val, train_bs, val_bs, img_size;
                               n_classes=1000)

Create `DataLoader`s for a tiny synthetic ImageNet‑like dataset. No augmentation
is performed – the images are raw random tensors.
"""
function construct_toy_dataloaders(
    n_train::Int, n_val::Int,
    train_bs::Int, val_bs::Int,
    img_size::Int; n_classes::Int=1000, dev=gpu_device()
)
    dims = (img_size, img_size, 3)
    train_ds = SimpleToyDataset(n_train, n_classes, dims, Random.GLOBAL_RNG)
    val_ds   = SimpleToyDataset(n_val,   n_classes, dims, Random.GLOBAL_RNG)
    train_dl = DataLoader(train_ds; batchsize=train_bs, shuffle=true, collate=true)
    val_dl   = DataLoader(val_ds;   batchsize=val_bs,   shuffle=false, collate=true)
    return dev(train_dl), dev(val_dl)
end

"""
    toy_imagenet_data(...)

Convenient wrapper that mirrors the signature of the full `imagenet_data`
helper, returning `(train_loader, val_loader, test_loader)`.
"""
function toy_imagenet_data(
    path_placeholder, train_bs::Int, val_bs::Int,
    img_size::Int; n_classes::Int=1000, dev=gpu_device(), n_val=1000, n_train=1000

)
    train_dl, val_dl = construct_toy_dataloaders(
        n_train, n_val, train_bs, val_bs, img_size; n_classes=n_classes, dev=dev)
    return train_dl, val_dl, nothing
end