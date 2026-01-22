"""
    SimpleToyDataset(n_samples, n_classes, image_dims; rng=Random.GLOBAL_RNG)

A minimal in‑memory dataset that yields random images of shape `image_dims`
(H×W×C) and one‑hot class labels. No file I/O, no complex augmentation
pipeline.
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
    construct_toy_dataloaders(...; backend=nothing)

Create `DataLoader`s for a tiny synthetic ImageNet‑like dataset.
If `backend` is provided (e.g., NCCLBackend), the datasets are wrapped
in DistributedDataContainer for proper DDP data sharding.
"""
function construct_toy_dataloaders(
    n_train::Int, n_val::Int,
    train_bs::Int, val_bs::Int,
    img_size::Int; n_classes::Int=1000,
    dev=gpu_device(),
    backend=nothing  # Pass the initialized backend here
)
    dims = (img_size, img_size, 3)

    train_ds = SimpleToyDataset(n_train, n_classes, dims, Random.GLOBAL_RNG)
    val_ds   = SimpleToyDataset(n_val,   n_classes, dims, Random.GLOBAL_RNG)

    # ── Distributed wrapping ────────────────────────────────────────
    # If backend is provided, wrap datasets for distributed training
    if !isnothing(backend)
        train_ds = DistributedDataContainer(backend, train_ds)
        val_ds   = DistributedDataContainer(backend, val_ds)
    end
    # ────────────────────────────────────────────────────────────────────

    train_dl = DataLoader(train_ds;
        batchsize=train_bs, shuffle=true, collate=true)
    val_dl   = DataLoader(val_ds;
        batchsize=val_bs, shuffle=false, collate=true)

    return dev(train_dl), dev(val_dl)
end

# """
#     toy_imagenet_data(...; backend=nothing)

# Convenient wrapper that mirrors the signature of the full `imagenet_data`
# helper, returning `(train_loader, val_loader, test_loader)`.
# """
# function toy_imagenet_data(
#     path_placeholder, train_bs::Int, val_bs::Int,
#     img_size::Int; n_classes::Int=1000,
#     dev=gpu_device(),
#     n_val=1000, n_train=1000,
#     backend=nothing  # Pass backend instead of is_distributed flag
# )
#     train_dl, val_dl = construct_toy_dataloaders(
#         n_train, n_val, train_bs, val_bs, img_size;
#         n_classes=n_classes, dev=dev, backend=backend)

#     return train_dl, val_dl, nothing
# end
