begin
using Revise
include("../../HelperFunctions/generate_networks_and_data.jl")
include("layerwise_pruning.jl")
include("../../HelperFunctions/plot_networks_and_their_output.jl")
include("../../HelperFunctions/break_loop.jl")

# Layerwise pruning example script. Projected gradient descent implementation.

m,ps,st = generate_dense_network([2, 25, 25, 1]; seed=1, scaling=1)
tstate = Training.TrainState(m,ps,st,Adam(1e-3))

dataset_size = 3000
train_set = generate_dataset(Int64(dataset_size),
    Int64(dataset_size),
    m,
    ps,
    st;
    dtype = Float32,
    sigma = 1f-20,
    dev = cpu_device(),
    seed = 1)

dtype = Float32
end

plot_weights(tstate)

input = [batch[1] for batch in train_set]

tstate, loglogs, layer_masks = layerwise_reverse_pruning(tstate, input; dtype = dtype, lr = dtype(0.01), alpha = dtype(1.025), dev = cpu_device(), epochs=50000, verbose=true, smoothing_window=400)

# tstate, loglogs, layer_masks = layerwise_pruning(tstate, input; dtype = dtype, lr = dtype(0.01), alpha = dtype(0.025), dev = cpu_device(), epochs=500000, verbose=true, smoothing_window=400, mask_start_value=dtype(1))

# # begin
PlotlyJS.plot(loglogs[2]["epochs"][1:end],loglogs[2]["train_loss"][1:end])
is_saturated(loglogs[2]["train_loss"], 400; plot_graph=true)

plot_weights(tstate)

# sq_error = dtype(0)
# for d in train_set
#     new_output = tstate.model(d[1], tstate.parameters, tstate.states)[1]
#     sq_error += sum((new_output .- d[2]).^2)
# end
# sq_error = sq_error / dtype(length(train_set))
# mean_sq_error = sq_error / dtype(size(train_set[1][1])[2])
