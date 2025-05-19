using Lux, Optimisers, Printf, Random, Zygote
using ChainRulesCore: ignore_derivatives
using Accessors, Revise

include("custom_linear_probabilistic_model.jl")
include("../../HelperFunctions/break_loop.jl")
include("../../HelperFunctions/random_gradient_pruning.jl")

"""
    prune_layer(input, output, weight, bias; dtype = Float32, lr = dtype(0.01), alpha = dtype(1.0), dev = cpu_device(), epochs=500, verbose=false, rng=Random.default_rng(), smoothing_window=400, mask_start_value = dtype(0), saturation_counter_max_value=3, p_saturation_counter_max_value=3, use_gauss_loss=true, mask=nothing, initial_sigma=[1f-1])

    This function prunes a single neural network layer by employing the Probabilistic Exact Gradient Pruning (PEGP) method. It returns a tuple (tps_p, logs), where tps_p are the pruned layer parameters and logs contain training loss curves. To keep the mask value between 0 and 1, the values of the mask are projected to [0,1] after each gradient update.
    
    Arguments:
        - `input`: Input activations from previous layer.
        - `output`: Outpt activations of present layer.
        - `weight`: Weights of present layer.
        - `bias`: Biases of present layer.
        - `lr`: Learning rate for layerwise pruning.
        - `alpha`: regularization strength.
        - `dev`: Device (either `Lux.cpu_device()` or `Lux.gpu_device()`)
        - `epochs`: Maximum number of training epochs.
        - `verbose`: Whether log information is printed or not.
        - `rng`: Random number generator.
        - `smoothing_window`: The smoothing_window is a number that determines how many epochs of the loss curve should be taken into account when determining convergence.
        - `mask_start_value`: The initial PEGP mask value. Default is 0.
        - `saturation_counter_max_value`: Number of times saturation must trigger before convergence is triggered.
        - `p_saturation_counter_max_value`: Number of times all mask values must be recorded to be either 0 or 1 before convergence is triggered.
        - `use_gauss_loss`: whether Gauss loss is used or not.
        - `mask`: Whether an initial mask is provided for the layerwise optimization. If not, then mask is initialized with all values equal to `mask_start_value`.
        - `initial_sigma`: If Gauss loss is used, then `initial_sigma` is the starting value for sigma in the loss.
"""
function prune_layer(input, output, weight, bias; dtype = Float32, lr = dtype(0.01), alpha = dtype(1.0), dev = cpu_device(), epochs=500, verbose=false, rng=Random.default_rng(), smoothing_window=400, mask_start_value = dtype(0), saturation_counter_max_value=3, p_saturation_counter_max_value=3, use_gauss_loss=true, mask=nothing, initial_sigma=[1f-1])
    
    function loss_fun(model::Prolinear, ps::NamedTuple, st::NamedTuple, (inp,outp)::Tuple)
        output_p, st = model(inp,ps,st)
        usual_loss = (outp .- output_p).^2
        pi_contribution = (ps.w.^2 .* ps.pw .* (1 .- ps.pw)) * (inp.^2) .+ ps.b.^2 .* ps.pb .* (1 .- ps.pb)
        loss = sum(usual_loss .+ pi_contribution) + alpha*(sum(ps.pw)+sum(ps.pb))
        stats = []
        return loss, st, stats
    end

    function loss_std(model::Prolinear, ps::NamedTuple, st::NamedTuple, (input,output)::Tuple)
        output_p, st = model(input,ps,st)
        usual_loss = (output .- output_p).^2
        # pi_contribution = (ps.w.^2 .* ps.pw .* (1 .- ps.pw)) * (input.^2) .+ ps.b.^2 .* ps.pb .* (1 .- ps.pb)
        loss = sum(usual_loss) # .+ pi_contribution)
        stats = []
        return loss, st, stats
    end
    function round_p!(w)
        w[w.>0.5].=1
        w[w.<=0.5].=0
    end

    function gauss_loss(model::Prolinear, ps::NamedTuple, st::NamedTuple, (inp,outp)::Tuple)
        p = ps.p
        sigma = sum(ps.sigma)
        output_p, st = model(inp,p,st)
        usual_loss = (outp .- output_p).^2
        pi_contribution = (p.w.^2 .* p.pw .* (1 .- p.pw)) * (inp.^2) .+ p.b.^2 .* p.pb .* (1 .- p.pb)

        num_samples_in_batch = dtype(size(inp)[2])

        loss = sum(usual_loss .+ pi_contribution)/(dtype(2)*sigma^2*log(dtype(2))) + dtype(0.5)*num_samples_in_batch*log2(2*pi*sigma^2) + alpha*(sum(p.pw)+sum(p.pb))

        stats = []
        return loss, st, stats
    end

    in_dim = size(weight)[2]
    out_dim = size(weight)[1]

    function almost_ones(a,rng,b)
        return a.* Lux.ones32(rng,b)
    end
    function almost_ones(a,rng,b,c)
        return a.* Lux.ones32(rng,b,c)
    end
    # model = Prolinear(in_dim, out_dim)
    # model = Prolinear(in_dim, out_dim; initial_pw_params=Lux.ones32, initial_pb_params=Lux.ones32)
    
    model = Prolinear(in_dim, out_dim; initial_pw_params=(rng,b,c)->almost_ones(mask_start_value,rng,b,c), initial_pb_params=(rng,b)->almost_ones(mask_start_value,rng,b))
    
    ps, st = Lux.setup(rng, model) |> dev
    ps.w .= weight
    ps.b .= bias
    if !isnothing(mask) # transfer masks of masked models
        ps.pw .= mask.weight
        ps.pb .= mask.bias
    end
    ps.pw[ps.w.==0].=0
    ps.pb[ps.b.==0].=0


    logs = Dict{String, Any}(
        "epochs" => Int[],
        "train_loss" => dtype[]
    )

    function clip_mask_p!(m)
        clamp!(m.p.pw,0,1)
        clamp!(m.p.pb,0,1)
        # m.p.pw[m.p.pw .< 0] .= 0
        # m.p.pw[m.p.pw .> 1] .= 1
        # m.p.pb[m.p.pb .< 0] .= 0
        # m.p.pb[m.p.pb .> 1] .= 1
    end
    function clip_mask_no_p!(m)
        clamp!(m.pw,0,1)
        clamp!(m.pb,0,1)
        # m.pw[m.pw .< 0] .= 0
        # m.pw[m.pw .> 1] .= 1
        # m.pb[m.pb .< 0] .= 0
        # m.pb[m.pb .> 1] .= 1
    end
    function p_0_1_p(w)
        return all(x -> (x==0.0 || x==1.0), w.p.pw) && all(x -> (x==0.0 || x==1.0), w.p.pb)
    end
    function p_0_1_no_p(w)
        return all(x -> (x==0.0 || x==1.0), w.pw) && all(x -> (x==0.0 || x==1.0), w.pb)
    end

    if use_gauss_loss
        ps = (sigma=initial_sigma, p=ps)
        loss_fctn = gauss_loss
        clip_mask! = clip_mask_p!
        p_0_1 = p_0_1_p
    else
        loss_fctn = loss_fun
        clip_mask! = clip_mask_no_p!
        p_0_1 = p_0_1_no_p
    end
    last_loss = dtype(0)

    opt = Optimisers.Adam(lr)
    
    tstate_p = Training.TrainState(model, ps, st, opt)
    vjp = AutoZygote()

    data = zip(input,output)
    saturation_counter = 0
    p_saturation_counter = 0

    for epoch in 1:epochs
        avg_loss = dtype(0)
        for batch in data
            grads, loss, stats, tstate_p = Training.single_train_step!(vjp, loss_fctn, batch, tstate_p)
            avg_loss += loss
            # projected gradient descent:
            ignore_derivatives() do
                clip_mask!(tstate_p.parameters)
            end
        end
        avg_loss = avg_loss / dtype(length(data))
        
        # Logging
        push!(logs["epochs"], epoch)
        push!(logs["train_loss"], avg_loss)

        if verbose
            if epoch % 50 == 1
                @printf "\rEpoch: %3d \t L0-Loss: %.5g\t" epoch avg_loss
            end
        end

        # convergenceCriteria:
        if epoch % 100 == 0 # check every 100 epochs
            
            p_either_0_or_1 = p_0_1(tstate_p.parameters)

            if p_either_0_or_1
                p_saturation_counter += 1
            else
                p_saturation_counter = 0
            end
            
            if is_saturated(logs["train_loss"], smoothing_window)
                saturation_counter += 1
            else
                saturation_counter = 0
            end

            if saturation_counter >= saturation_counter_max_value
                if verbose
                    # println("\nReached saturation value at epoch ", epoch)
                end
                break
            end
            if p_saturation_counter >= p_saturation_counter_max_value
                if verbose
                    # println("\nReached p-saturation value at epoch ", epoch)
                end
                break
            end
            if p_either_0_or_1
                if saturation_counter >= 1
                    if verbose
                        println("\nConverged at epoch ", epoch)
                    end
                    break
                end
            end
        end
        if epoch == epochs
            if verbose
                println("\nReached final epoch ", epoch)
            end
            break
        end
           
        last_loss = avg_loss
    end
    

    if use_gauss_loss
        tps_p = tstate_p.parameters.p
        if verbose
            # println("Final sigma ", tstate_p.parameters.sigma)
        end
    else
        tps_p = tstate_p.parameters
    end
    if verbose
        final_loss = dtype(0)
        for batch in data
            final_loss += loss_std(model, tps_p, tstate_p.states, batch)[1]
        end
        final_loss = final_loss / dtype(length(data))
        
        @printf "\nFinal L0-Loss: %.5g \t Loss: %.5g\n" last_loss final_loss
    end

    round_p!(tps_p.pw)
    round_p!(tps_p.pb)

    return tps_p, logs
end



"""
    layerwise_reverse_pruning(tstate, input; dtype = Float32, lr = dtype(0.01), alpha = dtype(1.0), dev = cpu_device(), epochs=500, verbose=false, smoothing_window=400, mask_start_value=dtype(0))

    This function prunes a given neural network by employing the Probabilistic Exact Gradient Pruning (PEGP) method layer by layer. It returns a tuple (tstate, logs), where tstate encodes network model and parameters and logs contain training loss curves. The function performs reverse pruning, starting from the last layer and applying random gradient pruning after each layer pruning to eliminate spurious weights. In this way, part of the layerwise pruning information is propagated through the entire network, making layerwise pruning overall more efficient.
    
    Arguments:
        - `tstate`: Lux TrainState, encoding neural network model and parameters.
        - `input`: Input data of first layer.
        - `dtype`: DataType, usually Float32.
        - `lr`: Learning rate for layerwise pruning.
        - `alpha`: regularization strength.
        - `dev`: Device (either `Lux.cpu_device()` or `Lux.gpu_device()`)
        - `epochs`: Maximum number of training epochs.
        - `verbose`: Whether log information is printed or not.
        - `smoothing_window`: The smoothing_window is a number that determines how many epochs of the loss curve should be taken into account when determining convergence.
        - `mask_start_value`: The initial PEGP mask value. Default is 0.
"""
function layerwise_reverse_pruning(tstate, input; dtype = Float32, lr = dtype(0.01), alpha = dtype(1.0), dev = cpu_device(), epochs=500, verbose=false, smoothing_window=400, mask_start_value=dtype(0))
    
    if verbose
        println("\n\nLayerwise pruning...")
    end
    if haskey(tstate.parameters,:p)
        tps = tstate.parameters.p
        if haskey(tstate.parameters.p[1],:sigma)
            initial_sigma = tstate.parameters.sigma
            use_gauss_loss = true
        else
            initial_sigma = [dtype(1f-1)] |> dev
            use_gauss_loss = false
        end
    else
        tps = tstate.parameters
        initial_sigma = [dtype(1f-1)] |> dev
        use_gauss_loss = false
    end

    # layer_masks = []
    loglogs = []

    if !haskey(tstate.parameters.p.layer_1, :weight)
        layer_output = []
        for (i,inp) in enumerate(input)
            r = tstate.model.layers.layer_1(inp, tstate.parameters.p.layer_1, tstate.states.st.layer_1)[1]
            push!(layer_output, r)
            @reset input[i] = r
        end
    else
        layer_output = deepcopy(input)
    end

    random_loss = Lux.MSELoss()
    data_size = [size(input[1]), (tstate.model.layers[end].out_dims, size(input[1])[2])]

    for layer_ind in length(tps):-1:1

        layer = tps[layer_ind]

        if !haskey(layer, :weight)
            continue
        end

        layer_input = deepcopy(input)
        for ind in eachindex(input)
            for (l_ind, l) in enumerate(tps)
                if !haskey(l, :weight) # <<--
                    continue
                elseif l_ind > 1 && !(typeof(tstate.model.layers[l_ind-1]) <: FlattenLayer) # haskey(tstate.model.layers[l_ind-1], :activation)
                    l_act = tstate.model.layers[l_ind-1].activation
                    layer_input[ind] = l_act.(layer_output[ind])
                end
                layer_output[ind] = l.weight * layer_input[ind] .+ l.bias
                if l_ind == layer_ind
                    break
                end
            end
        end
        if tstate.model.name == "masked model"
            mask = (weight = tstate.states[layer_ind].weight_mask, bias=tstate.states[layer_ind].bias_mask)
        elseif tstate.model.name == "PMMP model" && !haskey(tstate.parameters, :pp)
            mask = (weight = layer.weight_p, bias=layer.bias_p)
        elseif haskey(tstate.parameters, :pp)
            mask = (weight = tstate.parameters.pp[layer_ind].weight, bias = tstate.parameters.pp[layer_ind].bias)
        else
            mask = nothing
        end

        pruned_params, logs = prune_layer(layer_input, layer_output, layer.weight, layer.bias; dtype = dtype, lr = lr, alpha = alpha, dev = dev, epochs=epochs, verbose=verbose, smoothing_window=smoothing_window, mask_start_value=mask_start_value, use_gauss_loss=use_gauss_loss, mask=mask, initial_sigma=initial_sigma)

        if tstate.model.name == "masked model"
            tstate.states[layer_ind].weight_mask .= pruned_params.pw
            tstate.states[layer_ind].bias_mask .= pruned_params.pb
        elseif tstate.model.name == "PMMP model" && !haskey(tstate.parameters, :pp)
            tps[layer_ind].weight_p .= pruned_params.pw
            tps[layer_ind].bias_p .= pruned_params.pb
        elseif haskey(tstate.parameters, :pp)
            tstate.parameters.pp[layer_ind].weight .= pruned_params.pw
            tstate.parameters.pp[layer_ind].bias .= pruned_params.pb
        else
            mask = nothing
        end
        push!(loglogs, logs)

        w = pruned_params.w .* pruned_params.pw
        b = pruned_params.b .* pruned_params.pb

        # push!(layer_masks, (pruned_params.pw, pruned_params.pb))

        tps[layer_ind].weight .= w
        if Lux.has_bias(tstate.model.layers[layer_ind])
            tps[layer_ind].bias .= b
        end

        if verbose
            println("Finished layer ", layer_ind)
            # println("Pruned weight:")
            # println(pruned_params.pw)
            # println("\nPruned bias")
            # println(pruned_params.pb)
            # layer_output = layer_activation.(layer_linear_output)
            # diff = sum(abs.(layer_output-layer_input))
            # println("\nFit error: ", logs["train_loss"][end],"\n\n")
        end

        if haskey(tstate.parameters, :p)
            @reset tstate.parameters.p = tps
        else
            @reset tstate.parameters = tps
        end

        # plot_weights(tstate)
        if !(typeof(tstate.model.layers[1]) <: FlattenLayer)
            prune_with_random_gradient!(tstate, data_size, random_loss; dev=dev)
        end
        # plot_weights(tstate)

        if haskey(tstate.parameters,:p)
            tps = tstate.parameters.p
        else
            tps = tstate.parameters
        end
    end
    if verbose
        println("Layerwise pruning finished.")
    end

    return tstate, loglogs #, layer_masks
end
