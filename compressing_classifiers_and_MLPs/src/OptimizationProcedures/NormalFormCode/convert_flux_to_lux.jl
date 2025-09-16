import Flux
using Adapt, Lux, Random, Revise, Optimisers

function convert_to_lux(flux_model; copy_params=true, dev=cpu_device())
    flux_params = Flux.params(flux_model)
    lux_model = adapt(FromFluxAdaptor(), flux_model)
    lux_params, st = Lux.setup(Random.default_rng(), lux_model) |> dev
    if copy_params
        transfer_params!(flux_params, lux_params)
    end
    return lux_model, lux_params, st
end

function convert_to_tstate(flux_model; copy_params=true, dev=cpu_device(), opt = Optimisers.Adam, learning_rate = 1f-2)
    lux_model, lux_params, st = convert_to_lux(flux_model; copy_params=copy_params, dev=dev)
    tstate = Training.TrainState(lux_model, lux_params, st, opt(learning_rate))
    return tstate
end

function transfer_params!(flux_params, lux_params)
    i = 1
    for layer in lux_params
        for layer_component_value in layer
            layer_component_value .= flux_params[i]
            i += 1
        end
    end
    return
end

function convert_to_flux!(flux_model_with_lux_model_architecture, lux_params)
    fps = Flux.params(flux_model_with_lux_model_architecture)
    i = 1
    for layer in lux_params
        for layer_component_value in layer
            fps[i] .= layer_component_value
            i += 1
        end
    end
    return
end
function convert_fully_connected_network_to_flux(lux_params,lux_model)
    layers = []
    for l in lux_model.layers
        usebias = l.use_bias==Lux.static(true)
        push!(layers, Flux.Dense(l.in_dims => l.out_dims, l.activation; bias=usebias))
    end
    flux_model = Flux.Chain(layers...)

    fps = Flux.params(flux_model)
    i = 1
    for layer in lux_params
        for layer_component_value in layer
            fps[i] .= layer_component_value
            i += 1
        end
    end
    return flux_model
end

### Example usage:
# flux_model = Flux.Chain(Flux.Dense(2 => 3, relu), Flux.Dense(3 => 2));

# lux_model, lux_params, st = convert_to_lux(flux_model);

# Flux.params(flux_model)
# lux_params

# x = randn(Float32, 2, 32);
# o1 = flux_model(x)
# o2 = lux_model(x,lux_params,st)[1]
# o1 == o2
