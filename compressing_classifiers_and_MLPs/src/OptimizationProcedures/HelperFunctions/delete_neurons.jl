using Accessors, Revise

"""
    delete_neurons_lux!(state::Lux.Training.TrainState, loss_fun; reassign=false)

    Recursively deletes neurons in multi-layer-perceptrons that do not contribute to the computation of the function of the neural network anymore. Those are the neurons that do not have any incoming connections anymore. They usually arise from a previous pruning step.
    
    The process additionally removes biases of neurons without incoming connections and transfers the biases to the biases of subsequent layers by multiplying them with the outgoing connections and thereafter applying an activation function. This results in additional pruning of the network.

    Returns tstate (containing all model, optimizer and network parameters), loss_fun (the loss function).
"""
function delete_neurons_lux!(state::Lux.Training.TrainState, loss_fun; reassign=false)
    # when setting reassign=true, it is recommended to call the function with reassignment like this: tstate = delete_neurons_lux!(tstate; reassign=true). Otherwise, if you call it like this: tstate2 = delete_neurons_lux!(tstate; reassign=true), so that tstate is not re-assigned to the return value of delete_neurons_lux!(tstate; reassign=true), while reassign is true, then this modifies tstate only partially because the @reset-commands in the function below are ignored while the reassignment of values to matrices modifies tstate.
    # Hence you should either call tstate = delete_neurons_lux!(tstate; reassign=true), which can increase speed in training loops because it foregoes creating a copy of tstate,
    # or you should call tstate2 = delete_neurons_lux!(tstate; reassign=false), which prevents tstate to be modified but then requires the creation of a copy.
    
    function update_optimizer_state!(opt_state, l_name, mask, col_num)
        for (w_name,w) in pairs(opt_state[Symbol(l_name)])
            if !isnothing(w.state)
                for (ind,m) in enumerate(w.state)
                    if col_num == 1
                        if isa(m, AbstractArray{T,2} where T)
                            @reset opt_state[Symbol(l_name)][Symbol(w_name)].state[ind] = m[mask,:]
                        elseif isa(m, AbstractArray{T,1} where T)
                            @reset opt_state[Symbol(l_name)][Symbol(w_name)].state[ind] = m[mask]
                        end
                    elseif col_num == 2
                        if isa(m, AbstractArray{T,2} where T)
                            @reset opt_state[Symbol(l_name)][Symbol(w_name)].state[ind] = m[:,mask]
                        end
                    end
                end
            end
        end
        return opt_state
    end
    if !reassign 
        tstate = deepcopy(state)
    else
        tstate = state
    end
    if haskey(tstate.parameters,:p)
        tps = tstate.parameters.p
        tos = tstate.optimizer_state.p
    else
        tps = tstate.parameters
        tos = tstate.optimizer_state
    end
    i = 0
    biases = []
    biases_w = []
    biases_pp = []
    # biases_u = []
    out_dims = 0
    if typeof(tstate.model.layers[1]) <: FlattenLayer # !haskey(tstate.model.layers.layer_1, :activation)
        activation_of_previous_layer = tstate.model.layers.layer_2.activation
    else
        activation_of_previous_layer = tstate.model.layers.layer_1.activation
    end

    for (l_name, l) in pairs(tps)

        if !haskey(l, :weight)
            continue
        end

        i += 1
        if i>1 # this uses values from the previous iteration to modify the present layer
            @reset tstate.model.layers[Symbol(l_name)].in_dims = out_dims
            if haskey(tstate.parameters, :pw)
                tstate.parameters.pw[Symbol(l_name)].bias .+= (tstate.parameters.pw[Symbol(l_name)].weight[:, biases_w[1]] .* tstate.parameters.pp[Symbol(l_name)].weight[:, biases_pp[1]]) * activation_of_previous_layer.(biases_w[2] .* biases_pp[2])
            end
            # if haskey(tstate.parameters, :pp)
            #     tstate.parameters.pp[Symbol(l_name)].bias .+= tstate.parameters.pp[Symbol(l_name)].weight[:, biases_pp[1]] * activation_of_previous_layer.(biases_pp[2])
            # end
            # if haskey(tstate.parameters, :u)
            #     if isa(tstate.parameters.u, NamedTuple)
            #         @reset tstate.optimizer_state.u = update_optimizer_state!(tstate.optimizer_state.u, l_name, row_mask, 1)
            #         @reset tstate.optimizer_state.u = update_optimizer_state!(tstate.optimizer_state.u, l_name, row_mask, 1)
            #     end
            # end
            if haskey(tstate.states, :mask)
                tps[Symbol(l_name)].bias .+= (tps[Symbol(l_name)].weight[:, biases[1]] .* tstate.states.mask[Symbol(l_name)].weight[:, biases[1]]) * activation_of_previous_layer.(biases[2])
            else
                tps[Symbol(l_name)].bias .+= tps[Symbol(l_name)].weight[:, biases[1]] * activation_of_previous_layer.(biases[2])
            end

            exclude_cols = biases[1]
            column_mask = Int.(setdiff(1:size(tps[Symbol(l_name)].weight, 2), exclude_cols))
            # update parameters
            @reset tps[Symbol(l_name)].weight = tps[Symbol(l_name)].weight[:,column_mask]
            if tstate.model.name == "masked model"
                @reset tstate.states[Symbol(l_name)].weight_mask = tstate.states[Symbol(l_name)].weight_mask[:,column_mask]
            end
            if haskey(tstate.states, :mask)
                @reset tstate.states.mask[Symbol(l_name)].weight = tstate.states.mask[Symbol(l_name)].weight[:,column_mask]
            end
            if haskey(tstate.parameters, :pw) && haskey(tstate.parameters, :pp)
                @reset tstate.parameters.pw[Symbol(l_name)].weight = tstate.parameters.pw[Symbol(l_name)].weight[:,column_mask]
                @reset tstate.parameters.pp[Symbol(l_name)].weight = tstate.parameters.pp[Symbol(l_name)].weight[:,column_mask]

                if haskey(tstate.parameters, :u)
                    if isa(tstate.parameters.u, NamedTuple)
                        @reset tstate.parameters.u[Symbol(l_name)].weight = tstate.parameters.u[Symbol(l_name)].weight[:,column_mask]
                    end
                end
                if hasproperty(loss_fun, :grad_template)
                    grad_template = loss_fun.grad_template
                    @reset grad_template.p[Symbol(l_name)].weight = grad_template.p[Symbol(l_name)].weight[:,column_mask]
                    @reset grad_template.pw[Symbol(l_name)].weight = grad_template.pw[Symbol(l_name)].weight[:,column_mask]
                    @reset grad_template.pp[Symbol(l_name)].weight = grad_template.pp[Symbol(l_name)].weight[:,column_mask]
                    if haskey(tstate.parameters, :u)
                        if isa(tstate.parameters.u, NamedTuple)
                            @reset grad_template.u[Symbol(l_name)].weight = grad_template.u[Symbol(l_name)].weight[:,column_mask]
                        end
                    end
                    loss_fun.grad_template = grad_template
                end
            end
            if tstate.model.name == "PMMP model" && !haskey(tstate.parameters, :pp)
                @reset tps[Symbol(l_name)].weight_w = tps[Symbol(l_name)].weight_w[:,column_mask]
                @reset tps[Symbol(l_name)].weight_p = tps[Symbol(l_name)].weight_p[:,column_mask]
                @reset tps[Symbol(l_name)].u_w = tps[Symbol(l_name)].u_w[:,column_mask]
            end

            # update the optimizer state
            tos = update_optimizer_state!(tos, l_name, column_mask, 2)
            if haskey(tstate.parameters, :pw) && haskey(tstate.parameters, :pp)
                @reset tstate.optimizer_state.pw = update_optimizer_state!(tstate.optimizer_state.pw, l_name, column_mask, 2)
                @reset tstate.optimizer_state.pp = update_optimizer_state!(tstate.optimizer_state.pp, l_name, column_mask, 2)
            end
            if haskey(tstate.parameters, :u)
                if isa(tstate.parameters.u, NamedTuple)
                    @reset tstate.optimizer_state.u = update_optimizer_state!(tstate.optimizer_state.u, l_name, column_mask, 2)
                end
            end
        end
        row_mask = ones(Bool,size(tps[Symbol(l_name)].weight)[1])
        bias_inds = Int[]
        for (row_index, row) in enumerate(eachrow(tps[Symbol(l_name)].weight))
            if all(row .== 0)
                row_mask[row_index] = 0
                push!(bias_inds, row_index)
            end
        end
        if haskey(tstate.states, :mask)
            biases = (bias_inds, tps[Symbol(l_name)].bias[bias_inds] .* tstate.states.mask[Symbol(l_name)].bias[bias_inds])
        else
            biases = (bias_inds, tps[Symbol(l_name)].bias[bias_inds])
        end
        if haskey(tstate.parameters, :pw)
            biases_w = (bias_inds, tstate.parameters.pw[Symbol(l_name)].bias[bias_inds])
        end
        if haskey(tstate.parameters, :pp)
            biases_pp = (bias_inds, tstate.parameters.pp[Symbol(l_name)].bias[bias_inds])
        end
        # if haskey(tstate.parameters, :u)
        #     if isa(tstate.parameters.u, NamedTuple)
        #         biases_u = (bias_inds, tstate.parameters.u[Symbol(l_name)].bias[bias_inds])
        #     end
        # end
        out_dims = sum(Int.(row_mask))

        # update parameters
        @reset tps[Symbol(l_name)].weight = tps[Symbol(l_name)].weight[row_mask,:]
        @reset tps[Symbol(l_name)].bias = tps[Symbol(l_name)].bias[row_mask]
        
        @reset tstate.model.layers[Symbol(l_name)].out_dims = out_dims
        activation_of_previous_layer = tstate.model.layers[Symbol(l_name)].activation
        
        if haskey(tstate.states, :mask)
            @reset tstate.states.mask[Symbol(l_name)].weight = tstate.states.mask[Symbol(l_name)].weight[row_mask,:]
            @reset tstate.states.mask[Symbol(l_name)].bias = tstate.states.mask[Symbol(l_name)].bias[row_mask]
        end
        if tstate.model.name == "masked model"
            @reset tstate.states[Symbol(l_name)].weight_mask = tstate.states[Symbol(l_name)].weight_mask[row_mask,:]
            @reset tstate.states[Symbol(l_name)].bias_mask = tstate.states[Symbol(l_name)].bias_mask[row_mask]
        end
        if haskey(tstate.parameters, :pw) && haskey(tstate.parameters, :pp)
            # @reset tstate.parameters.p[Symbol(l_name)].weight = tstate.parameters.p[Symbol(l_name)].weight[row_mask,:]
            @reset tstate.parameters.pw[Symbol(l_name)].weight = tstate.parameters.pw[Symbol(l_name)].weight[row_mask,:]
            @reset tstate.parameters.pp[Symbol(l_name)].weight = tstate.parameters.pp[Symbol(l_name)].weight[row_mask,:]
            # @reset tstate.parameters.p[Symbol(l_name)].bias = tstate.parameters.p[Symbol(l_name)].bias[row_mask]
            @reset tstate.parameters.pw[Symbol(l_name)].bias = tstate.parameters.pw[Symbol(l_name)].bias[row_mask]
            @reset tstate.parameters.pp[Symbol(l_name)].bias = tstate.parameters.pp[Symbol(l_name)].bias[row_mask]
            if haskey(tstate.parameters, :u)
                if isa(tstate.parameters.u, NamedTuple)
                    @reset tstate.parameters.u[Symbol(l_name)].weight = tstate.parameters.u[Symbol(l_name)].weight[row_mask,:]
                    @reset tstate.parameters.u[Symbol(l_name)].bias = tstate.parameters.u[Symbol(l_name)].bias[row_mask]
                end
            end
            if hasproperty(loss_fun, :grad_template)
                grad_template = loss_fun.grad_template
                @reset grad_template.p[Symbol(l_name)].weight = grad_template.p[Symbol(l_name)].weight[row_mask,:]
                @reset grad_template.pw[Symbol(l_name)].weight = grad_template.pw[Symbol(l_name)].weight[row_mask,:]
                @reset grad_template.pp[Symbol(l_name)].weight = grad_template.pp[Symbol(l_name)].weight[row_mask,:]
                @reset grad_template.p[Symbol(l_name)].bias = grad_template.p[Symbol(l_name)].bias[row_mask]
                @reset grad_template.pw[Symbol(l_name)].bias = grad_template.pw[Symbol(l_name)].bias[row_mask]
                @reset grad_template.pp[Symbol(l_name)].bias = grad_template.pp[Symbol(l_name)].bias[row_mask]
                if haskey(tstate.parameters, :u)
                    if isa(tstate.parameters.u, NamedTuple)
                        @reset grad_template.u[Symbol(l_name)].weight = grad_template.u[Symbol(l_name)].weight[row_mask,:]
                        @reset grad_template.u[Symbol(l_name)].bias = grad_template.u[Symbol(l_name)].bias[row_mask]
                    end
                end
                loss_fun.grad_template = grad_template
            end
        end
        if tstate.model.name == "PMMP model" && !haskey(tstate.parameters, :pw)
            @reset tps[Symbol(l_name)].weight_w = tps[Symbol(l_name)].weight_w[row_mask,:]
            @reset tps[Symbol(l_name)].weight_p = tps[Symbol(l_name)].weight_p[row_mask,:]
            @reset tps[Symbol(l_name)].u_w = tps[Symbol(l_name)].u_w[row_mask,:]
            @reset tps[Symbol(l_name)].bias_w = tps[Symbol(l_name)].bias_w[row_mask]
            @reset tps[Symbol(l_name)].bias_p = tps[Symbol(l_name)].bias_p[row_mask]
            @reset tps[Symbol(l_name)].u_b = tps[Symbol(l_name)].u_b[row_mask]
        end
        if isempty(tps[Symbol(l_name)].weight)
            throw(ArgumentError("A weight matrix was completely deleted in the 'delete_neurons' function. This can lead to errors in the training process. Layer name: $l_name"))
        end

        # update the optimizer state
        tos = update_optimizer_state!(tos, l_name, row_mask, 1)
        if haskey(tstate.parameters, :pw) && haskey(tstate.parameters, :pp)
            @reset tstate.optimizer_state.pw = update_optimizer_state!(tstate.optimizer_state.pw, l_name, row_mask, 1)
            @reset tstate.optimizer_state.pp = update_optimizer_state!(tstate.optimizer_state.pp, l_name, row_mask, 1)
        end
        if haskey(tstate.parameters, :u)
            if isa(tstate.parameters.u, NamedTuple)
                @reset tstate.optimizer_state.u = update_optimizer_state!(tstate.optimizer_state.u, l_name, row_mask, 1)
            end
        end

    end
    if haskey(tstate.parameters,:p)
        @reset tstate.parameters.p = tps
        @reset tstate.optimizer_state.p = tos
    else
        @reset tstate.parameters = tps 
        @reset tstate.optimizer_state = tos
    end
    return tstate, loss_fun 
end


