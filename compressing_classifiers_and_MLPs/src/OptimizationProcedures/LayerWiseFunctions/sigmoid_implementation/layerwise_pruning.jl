using Lux, Optimisers, Printf, Random, Zygote
using ChainRulesCore: ignore_derivatives
using Accessors, Revise

includet("custom_linear_probabilistic_model.jl")
includet("../../HelperFunctions/break_loop.jl")


"""
    prune_layer(input, output, weight, bias; dtype = Float32, lr = dtype(0.01), alpha = dtype(1.0), dev = cpu_device(), epochs=500, verbose=false, rng=Random.default_rng(), smoothing_window=400, mask_precision=dtype(0.0004))

    This function prunes a single neural network layer by employing the Probabilistic Exact Gradient Pruning (PEGP) method. It returns a tuple of pruned layer parameters and logs (contain training loss curves). To keep the mask value between 0 and 1, a sigmoid function is employed.
    
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
function prune_layer(input, output, weight, bias; dtype = Float32, lr = dtype(0.01), alpha = dtype(1.0), dev = cpu_device(), epochs=500, verbose=false, rng=Random.default_rng(), smoothing_window=400, mask_precision=dtype(0.0004))

    in_dim = size(weight)[2]
    out_dim = size(weight)[1]
    model = Prolinear(in_dim, out_dim)

    ps, st = Lux.setup(rng, model) |> dev
    ps.w .= weight
    ps.b .= bias
    last_rounded_pw = copy(ps.pw)
    last_rounded_pb = copy(ps.pb)
    new_rounded_pw = copy(ps.pw)
    new_rounded_pb = copy(ps.pb)

    logs = Dict{String, Any}(
        "epochs" => Int[],
        "train_loss" => dtype[]
    )

    opt = Adam(lr)
    
    tstate_p = Training.TrainState(model, ps, st, opt)
    vjp = AutoZygote()
    
    function loss_fun(model::Prolinear, ps::NamedTuple, st::NamedTuple, (input,output)::Tuple)
        pws = fast_activation(sigmoid_fast,ps.pw)
        pbs = fast_activation(sigmoid_fast,ps.pb)

        output_p = (ps.w .* pws) * input .+ (ps.b .* pbs)
        usual_loss = (output .- output_p).^2

        pi_contribution = (ps.w.^2 .* pws .* (1 .- pws)) * (input.^2) .+ ps.b.^2 .* pbs .* (1 .- pbs)

        loss =  alpha*(sum(pws)+sum(pbs)) + sum(usual_loss) + sum(pi_contribution)
        
        stats = []
        return loss, st, stats
    end

    function loss_std(model::Prolinear, ps::NamedTuple, st::NamedTuple, (input,output)::Tuple)
        pws = fast_activation(sigmoid_fast,ps.pw)
        pbs = fast_activation(sigmoid_fast,ps.pb)

        output_p = (ps.w .* pws) * input .+ (ps.b .* pbs)
        usual_loss = (output .- output_p).^2
        loss = sum(usual_loss)
        stats = []
        return loss, st, stats
    end
    
    # convert sigmoid parameters to 0's and 1's:
    function round_p(w,mask_precision)
        w[w.>=1-mask_precision].=1
        w[w.<=mask_precision].=0
    end

    last_loss = dtype(0)
    data = zip(input,output)

    for epoch in 1:epochs
        avg_loss = dtype(0)
        for batch in data
            grads, loss, stats, tstate_p = Training.single_train_step!(vjp, loss_fun, batch, tstate_p)
            avg_loss += loss
        end
        avg_loss = avg_loss / dtype(length(data))
        
        # Logging
        push!(logs["epochs"], epoch)
        push!(logs["train_loss"], avg_loss)

        if verbose
            if epoch % 50 == 1
                @printf "Epoch: %3d \t L0-Loss: %.5g\n" epoch avg_loss
            end
        end

        # convergenceCriteria:
        if epoch % 100 == 0 # check every 100 epochs
            
            new_rounded_pw .= fast_activation(sigmoid_fast,tstate_p.parameters.pw)
            new_rounded_pb .= fast_activation(sigmoid_fast,tstate_p.parameters.pb)
            round_p(new_rounded_pw, mask_precision)
            round_p(new_rounded_pb, mask_precision)
            last_rounded_pw .= fast_activation(sigmoid_fast,last_rounded_pw)
            last_rounded_pb .= fast_activation(sigmoid_fast,last_rounded_pb)
            round_p(last_rounded_pw, mask_precision)
            round_p(last_rounded_pb, mask_precision)

            p_either_0_or_1 = all(x -> (x==0.0 || x==1.0), new_rounded_pw) && all(x -> (x==0.0 || x==1.0), new_rounded_pb)
            
            p_didnt_change = all(last_rounded_pw .== new_rounded_pw) && all(last_rounded_pb .== new_rounded_pb)

            if p_didnt_change && p_either_0_or_1 && is_saturated(logs["train_loss"], smoothing_window; plot_graph=true)
                if verbose
                    println("Converged at epoch ", epoch)
                end
                break
            end
        end
        if epoch == epochs
            if verbose
                println("Reached final epoch ", epoch)
            end
            break
        end
           
        last_rounded_pw .= tstate_p.parameters.pw
        last_rounded_pb .= tstate_p.parameters.pb
        
        last_loss = avg_loss
    end
    
    if verbose
        final_loss = dtype(0)
        for batch in data
            final_loss += loss_std(model, tstate_p.parameters, tstate_p.states, batch)[1]
        end
        final_loss = final_loss / dtype(length(data))
        
        @printf "Final L0-Loss: %.5g \t Loss: %.5g\n" last_loss final_loss
    end

    tstate_p.parameters.pw .= Lux.sigmoid.(tstate_p.parameters.pw)
    tstate_p.parameters.pb .= Lux.sigmoid.(tstate_p.parameters.pb)
    round_p(tstate_p.parameters.pw, mask_precision)
    round_p(tstate_p.parameters.pb, mask_precision)
    
    return tstate_p.parameters, logs
end


"""
    layerwise_pruning(tstate, input; dtype = Float32, lr = dtype(0.01), alpha = dtype(1.0), seed = 42, dev = cpu_device(), epochs=500, verbose=false, loss_change_threshold=dtype(1e-4), smoothing_window=400)

    This function prunes a given neural network by employing the Probabilistic Exact Gradient Pruning (PEGP) method (sigmoid implementation) layer by layer. It returns a tuple (tstate, logs), where tstate encodes network model and parameters and logs contain training loss curves.
    
    Arguments:
        - `tstate`: Lux TrainState, encoding neural network model and parameters.
        - `input`: Input data of first layer.
        - `dtype`: DataType, usually Float32.
        - `lr`: Learning rate for layerwise pruning.
        - `alpha`: regularization strength.
        - `seed`: A seed ensuring that the same initial network parameters are chosen everytime the function is run.
        - `dev`: Device (either `Lux.cpu_device()` or `Lux.gpu_device()`)
        - `epochs`: Maximum number of training epochs.
        - `verbose`: Whether log information is printed or not.
        - `loss_change_threshold`: A threshold that makes convergence more likely when set to higher values.
        - `smoothing_window`: The smoothing_window is a number that determines how many epochs of the loss curve should be taken into account when determining convergence.
"""
function layerwise_pruning(tstate, input; dtype = Float32, lr = dtype(0.01), alpha = dtype(1.0), seed = 42, dev = cpu_device(), epochs=500, verbose=false, loss_change_threshold=dtype(1e-4), smoothing_window=400)
    
    if haskey(tstate.parameters,:mu)
        tps = tstate.parameters.mu
    else
        tps = tstate.parameters
    end

    loglogs = []
    output = deepcopy(input)
    for (layer_ind, layer) in enumerate(tps)
        for (ind,batch) in enumerate(input)
            output[ind] = layer.weight * batch .+ layer.bias
        end
        
        pruned_params, logs = prune_layer(input, output, layer.weight, layer.bias; dtype = dtype, lr = lr, alpha = alpha, dev = dev, epochs=epochs, verbose=verbose, smoothing_window=smoothing_window)

        push!(loglogs, logs)

        w = pruned_params.w .* pruned_params.pw
        b = pruned_params.b .* pruned_params.pb

        tps[layer_ind].weight .= w
        if Lux.has_bias(tstate.model.layers[layer_ind])
            tps[layer_ind].bias .= b
        end

        layer_activation = tstate.model.layers[layer_ind].activation
        for (ind,batch) in enumerate(output)
            input[ind] = layer_activation.(batch)
        end

        if verbose
            println("\nFinished layer ", layer_ind)
            println("Pruned weight:")
            println(pruned_params.pw)
            println("\nPruned bias")
            println(pruned_params.pb)

            println("\nFit error: ", logs["train_loss"][end],"\n\n")
        end
    end

    if haskey(tstate.parameters, :mu)
        @reset tstate.parameters.mu = tps
    else
        @reset tstate.parameters = tps
    end

    return tstate, loglogs
end
