begin
using Revise
include("../../HelperFunctions/generate_networks_and_data.jl")
include("layerwise_pruning.jl")
include("../../HelperFunctions/plot_networks_and_their_output.jl")
include("../../HelperFunctions/break_loop.jl")

# Layerwise pruning example script. Sigmoid implementation.

m,ps,st = generate_dense_network([2, 25, 25, 1]; seed=35, scaling=2)
tstate = Training.TrainState(m,ps,st,Adam(1e-5))

dataset_size = 300
train_set = generate_dataset(Int64(dataset_size*10), 
    Int64(dataset_size), 
    m,
    ps, 
    st; 
    dtype = Float32,
    sigma = 0.01f0,
    dev = cpu_device(),
    seed = 1)

dtype = Float32
end

plot_weights(tstate)

input = [batch[1] for batch in train_set]
tstate, loglogs = layerwise_pruning(tstate, input; dtype = dtype, lr = dtype(0.0001), alpha = dtype(0.0), seed = 42, dev = cpu_device(), epochs=50000, verbose=true, loss_change_threshold=dtype(1e-4), smoothing_window=400)

ni = 1
PlotlyJS.plot(loglogs[ni]["epochs"][1:end],loglogs[ni]["train_loss"][1:end])
is_saturated(loglogs[ni]["train_loss"], 400; plot_graph=true)

plot_weights(tstate)

sq_error = dtype(0)
for d in train_set
    new_output = tstate.model(d[1], tstate.parameters, tstate.states)[1]
    sq_error += sum((new_output .- d[2]).^2)
end
sq_error = sq_error / dtype(length(train_set))
mean_sq_error = sq_error / dtype(size(train_set[1][1])[2])
