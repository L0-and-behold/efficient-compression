"""
Neural network normalization utilities for canonical form representation.

Provides functions to transform neural networks into a unique normal form by eliminating
redundancies from neuron permutations, trivial neurons, and sign symmetries. This enables
meaningful comparison of functionally equivalent networks.

# Normal Form Definition
A neural network is in normal form when:
1. **Trivial neurons deleted**: All-zero neurons are removed
2. **Bias-based sorting**: Hidden neurons sorted by |bias| (largest to smallest)  
3. **Weight-based tie-breaking**: Ambiguous neurons sorted by outgoing weight magnitudes
4. **Positive bias signs**: Signs flipped so all biases are positive (for odd activations)

# Key Functions
- `normal_form`: Transform network to canonical normal form
- `is_same`: Check functional equivalence of two networks
- `delete_zero_neurons`: Remove trivial all-zero neurons
- `sort_by_bias`: Sort neurons by absolute bias values
- `normalize_neuron_signs`: Ensure positive biases via sign flips
"""

"""
Transform a 2-hidden-layer neural network into canonical normal form.

Applies a sequence of transformations to create a unique representative of the 
equivalence class of functionally identical networks. The normal form eliminates
ambiguities from neuron permutations and sign symmetries.

# Arguments
- `nn::Flux.Chain`: Neural network with exactly 3 layers (2 hidden)
- `device`: Target device function (default: Flux.Flux.gpu)

# Returns
- `Flux.Chain`: Functionally equivalent network in normal form

# Transformation Steps
1. **Delete trivial neurons**: Remove all-zero parameter neurons
2. **Sort by bias magnitude**: Order hidden neurons by |bias| (descending)
3. **Resolve ties by weights**: Break bias ties using outgoing weight magnitudes
4. **Normalize signs**: Flip signs to ensure positive biases

# Notes
- Preserves network function while creating unique representation
- Enables meaningful comparison between functionally equivalent networks
- Assumes odd activation functions (tanh) for sign symmetry
"""
function normal_form(nn::Flux.Chain; device = Flux.gpu)::Flux.Chain
    nn = nn |> Flux.cpu

    normal_form_nn = delete_zero_neurons(nn) |> Flux.cpu
    normal_form_nn = sort_by_bias(normal_form_nn) |> Flux.cpu
    normal_form_nn = sort_nn_along_weights(normal_form_nn) |> Flux.cpu
    normal_form_nn = normalize_neuron_signs(normal_form_nn) |> Flux.cpu

    return normal_form_nn |> device
end

"""
Check if two neural networks are functionally equivalent.

Determines whether two networks can be transformed into each other through
neuron permutations and trivial neuron deletion by comparing their normal forms.

# Arguments
- `nn1::Flux.Chain`: First neural network
- `nn2::Flux.Chain`: Second neural network  
- `device`: Target device function (default: Flux.gpu)

# Returns
- `Bool`: True if networks compute the same function

# Process
1. Transform both networks to normal form
2. Compare all parameters element-wise
3. Return true if all parameters match exactly

# Notes
- Networks are functionally equivalent if they differ only by:
  - Hidden neuron permutations
  - Trivial (all-zero) neurons
  - Sign flips with odd activations
"""
function is_same(nn1::Flux.Chain, nn2::Flux.Chain; device=Flux.gpu)::Bool
    nn1 = nn1 |> Flux.cpu
    nn2 = nn2 |> Flux.cpu

    ann1 = deepcopy(nn1)
    ann2 = deepcopy(nn2)

    ann1 = normal_form(ann1; device=device)
    ann2 = normal_form(ann2; device=device)

    for (i, p) in enumerate(Flux.params(ann1))
        p2 = Flux.params(ann2)[i]
        if p != p2

            nn1 = nn1 |> device
            nn2 = nn2 |> device
            return false
        end
    end

    nn1 = nn1 |> device
    nn2 = nn2 |> device
    return true
end

"""
Print the weights and biases of a neural network in normal form.

Displays the normalized parameter matrices and vectors for inspection.

# Arguments
- `nn::Flux.Chain`: Neural network with exactly 3 layers (2 hidden)

# Output Format
For each layer:
- Weight matrix
- Bias vector
"""
function print_normal_form(nn::Flux.Chain)
    if length(nn.layers) != 3
        error("The network must have exactly 3 layers (2 hidden layers)")
    end
    
    nnn = deepcopy(nn) |> Flux.cpu
    nnn = normal_form(nnn; device=device) |> Flux.cpu

    for i in 1:3
        println("Layer $i \n")
        println("Weights")
        println(Flux.params(nnn.layers[i])[1])
        println("Biases")
        println(Flux.params(nnn.layers[i])[2], "\n")
    end
    println("---")
end

###
# Helper functions to bring a neural network to normal form
###

"""
Remove trivial (all-zero parameter) neurons from the network.

Identifies neurons with all parameters equal to zero and removes them along
with their connections, creating a smaller functionally equivalent network.

# Arguments
- `nn::Flux.Chain`: Neural network with exactly 3 layers (2 hidden)
- `device`: Target device function (default: Flux.gpu)

# Returns
- `Flux.Chain`: Network with trivial neurons removed

# Process
1. Identify zero rows in concatenated [weights, bias] matrices
2. Remove zero neurons from current layer
3. Remove corresponding connections from next layer
4. Construct smaller network with remaining parameters

# Notes
- Preserves network function by removing only inactive neurons
- May significantly reduce network size for sparse networks
"""
function delete_zero_neurons(nn::Flux.Chain; device=Flux.gpu)::Flux.Chain

    nn = nn |> Flux.cpu

    if length(nn.layers) != 3
        error("The network must have exactly 3 layers (2 hidden layers)")
    end

    W, b, θ = Dict(), Dict(), Dict()

    for i in 1:3
        W[i] = Flux.params(nn.layers[i])[1]
        b[i] = Flux.params(nn.layers[i])[2]
        θ[i] = hcat(W[i], b[i])
    end

    # delete zero neurons
    for layer in 1:3
        zero_columns_logical = [θ[layer][i, :] == [0.0 for i in 1:size(θ[layer])[2]] for i in 1:size(θ[layer])[1]]
        # delete zero rows in given layer
        W[layer] = W[layer][.!zero_columns_logical, :]
        b[layer] = b[layer][.!zero_columns_logical]
        # delete attached columns in next layer
        if layer < 3
            W[layer + 1] = W[layer + 1][:, .!zero_columns_logical]
        end
    end

    # create new network
    input_size = size(W[1])[2]
    hidden_1_size = size(W[1])[1]
    hidden_2_size = size(W[2])[1]
    output_size = size(W[3])[1]

    smaller_nn = get_random_teacher(input_size, hidden_1_size, hidden_2_size, output_size; device=device) |> Flux.cpu

    for layer in 1:3
        Flux.params(smaller_nn.layers[layer])[1] .= W[layer]
        Flux.params(smaller_nn.layers[layer])[2] .= b[layer]
    end

    nn = nn |> device

    return smaller_nn |> device
end

### sort by bias

"""
Sort hidden layer neurons by absolute bias values (largest to smallest).

Reorders neurons within each hidden layer based on |bias| while preserving
the computed function through appropriate weight matrix adjustments.

# Arguments
- `nn::Flux.Chain`: Neural network with exactly 3 layers
- `device`: Target device function (default: Flux.gpu)

# Returns
- `Flux.Chain`: Network with bias-sorted hidden neurons

# Process
1. Generate permutations based on descending |bias| order
2. Apply permutations using adjust_params helper function
3. Input and output layers remain unchanged

# Notes
- Creates partial ordering that may have ties for equal |bias| values
- Ties resolved by subsequent weight-based sorting
- Preserves network function through coordinated parameter reordering
"""
function sort_by_bias(nn::Flux.Chain; device = Flux.gpu)::Flux.Chain


    if length(nn.layers) != 3
        error("The network must have exactly 3 layers. Two hidden layers and an output later.")
    end

    new = deepcopy(nn) |> Flux.cpu


    """
    Generate permutations to sort neurons by descending |bias|.
        
    Returns dictionaries p (permutations) and q (inverse permutations)
    for each layer, with input and output layers using identity permutations.
    """
    function _get_bias_permutations(nn::Flux.Chain)
    
        p = Dict()
        q = Dict()
    
        # layer 0 - input does not get permutated
        a = size(Flux.params(nn.layers[1])[1])[2]
        p[0] = [x for x in 1:a]
        q[0] = invperm(p[0])
    
        # hidden layers 
        for i in 1:2
            b = Flux.params(nn.layers[i])[2] # bias vector
            p[i] = sortperm(-abs.(b))
            q[i] = invperm(p[i])
        end

        # output layer - no sorting of neurons as this would change the function computed by the network
        a = size(Flux.params(nn.layers[3])[2])[1]
        p[3] = [x for x in 1:a]
        q[3] = invperm(p[3])

        return p, q
    end
    
    p, q = _get_bias_permutations(nn)
    new = adjust_params(new, p, q)

    return new |> device
end

### sort by weights

"""
Resolve bias-sorting ambiguities using outgoing weight magnitudes.

For neurons with equal |bias| values, applies lexicographic ordering based on
outgoing weight magnitudes to create unique neuron ordering.

# Arguments
- `nn::Flux.Chain`: Neural network with bias-sorted neurons
- `device`: Target device function (default: Flux.gpu)

# Returns
- `Flux.Chain`: Network with weight-resolved neuron ordering

# Process
For each hidden layer:
1. Identify neurons with ambiguous bias ordering
2. Sort ambiguous neurons by outgoing weights (column by column)
3. Iteratively reduce ambiguity until unique ordering achieved

# Notes
- Only affects neurons with tied |bias| values
- Uses lexicographic ordering: first by weight to neuron 1, then neuron 2, etc.
- Output layer neurons never reordered to preserve function
"""
function sort_nn_along_weights(nn::Flux.Chain; device = Flux.gpu)::Flux.Chain
    # we do not sort the output layer as it would change the function computed by the network

    nn = nn |> Flux.cpu

    for i in 1:2
        nn = sort_layer_along_weights(nn, i; device=device)
    end

    return nn |> device
end

"""
Sort neurons in a single layer by outgoing weight magnitudes.

Resolves ambiguities from bias-based sorting by examining outgoing weight
columns in order, applying lexicographic tie-breaking.

# Arguments
- `nn::Flux.Chain`: Neural network to sort
- `layer_idx::Int`: Index of layer to sort (1 or 2 for hidden layers)
- `device`: Target device function (default: Flux.gpu)

# Returns
- `Flux.Chain`: Network with layer-specific weight-based sorting applied
"""
function sort_layer_along_weights(nn::Flux.Chain, layer_idx::Int; device = Flux.gpu)::Flux.Chain
    nn = nn |> Flux.cpu

    # initialize ambiguity_mask : identify the neurons that cannot be uniquely sorted by bias
    b = Flux.params(nn.layers[layer_idx])[2] # bias vector
    ambiguous_neurons = ambiguity_mask(b)

    W = Flux.params(nn.layers[layer_idx])[1] # weight matrix
    for i in 1:size(W, 2)
        w = W[:, i]

        nn, ambiguous_neurons = sort_along_a_weight_column(nn, ambiguous_neurons, layer_idx, i; device=device)

    end

    return nn |> device
end

"""
Sort ambiguous neurons by magnitude of weights in a specific column.

Applies sorting to neurons that remain ambiguous after previous sorting steps,
using weights from a specific outgoing connection.

# Arguments
- `nn::Flux.Chain`: Neural network to sort
- `ambiguous_neurons::Vector{Bool}`: Mask indicating which neurons are ambiguous
- `layer_idx::Int`: Layer containing neurons to sort
- `column_idx::Int`: Weight column to use for sorting
- `device`: Target device function (default: Flux.gpu)

# Returns
- `Tuple{Flux.Chain, Vector{Bool}}`: (sorted network, updated ambiguity mask)

# Process
1. Extract weight column for sorting
2. Generate permutation for ambiguous neurons only
3. Apply permutation while preserving non-ambiguous neuron positions
4. Update ambiguity mask for remaining ties
"""
function sort_along_a_weight_column(nn::Flux.Chain, ambiguous_neurons, layer_idx::Int, column_idx::Int; device = Flux.gpu)

    """
    Get the identity permutation for the neurons in a given multi-layer perceptron.
    """
    function _identity_permuation(nn::Flux.Chain)

        p = Dict()
        q = Dict()
        p[0] = [x for x in 1:size(Flux.params(nn.layers[1])[1])[2]]
        q[0] = invperm(p[0])
        for i in 1:3
            b = Flux.params(nn.layers[i])[2] # bias vector
            p[i] = [x for x in 1:length(b)]
            q[i] = invperm(p[i])
        end
        return p, q
    end    
    
    p, q = _identity_permuation(nn)
    
    new = deepcopy(nn) |> Flux.cpu
    W = Flux.params(new.layers[layer_idx])[1]
    w = W[:, column_idx] # the column of parameters we want to sort along

    p[layer_idx] = get_permutation(ambiguous_neurons, w)
    q[layer_idx] = invperm(p[layer_idx])
    new = adjust_params(new, p, q)
    
    ambiguous_neurons = Vector(ambiguous_neurons .& ambiguity_mask(w)) # the ambiguity mask is getting smaller

    new = new |> device
    return new, ambiguous_neurons
end


"""
Input:
    - vector : a vector of values
Output:
    - a vector of booleans that specifies which cannot be uniquely sorted along the input vector.
"""
function ambiguity_mask(vector)::Vector{Bool}
    ambiguous = [false for _ in vector]
    v_dic = Dict()
    for value in Set(vector)
        v_dic[value] = findall(x -> x == value, vector)
    end
    for key in keys(v_dic)
        if length(v_dic[key]) > 1
            ambiguous[v_dic[key]] .= true
        end
    end
    return ambiguous
end


"""
Input:
    - ambiguity_mask : a vector of booleans that specifies which neurons are ambiguous. E.g. use `ambiguity_mask(b)` to get this vector.
    - w : the vector of weights that we want to sort along. E.g. use `W[:, 1]` to get this vector.
Output:
    - a permutation of the neurons based on the magnitude of the weights specified by `w` while only considering the neurons that are ambiguous.
        the permutation is such that the neurons that are not ambiguous are kept in their original order.
"""
function get_permutation(ambiguous_neurons::Vector{Bool}, w::Vector)
    w_masked = [w[i] for i in 1:length(w) if ambiguous_neurons[i] == 1]
    perm = sortperm(-abs.(w_masked)) # permutation of the ambiguous neurons

    an = Dict() # look up table where the ambiguous neurons are
    j = 1
    for (i, x) in enumerate(ambiguous_neurons)
        if !x
            continue
        else
            an[j] = i
            j += 1
        end
    end

    perm .= [an[perm[i]] for i in 1:length(perm)]

    permu = []
    j = 1
    for i in 1:length(w)
        if !ambiguous_neurons[i]
            append!(permu, i)
        else
            append!(permu, perm[j])
            j += 1
        end
    end
    permu = Vector{Int64}(permu)

    return permu
end


"""
Flip the signs of parameters in the neural network such that all biases are positive and the neural network computes the same function.
    
Explanation: 
    For odd activation functions, such as tahn, the computed function of the nn does not change if the sign of all parameters attached to a neuron are flipped.
        w tanh( ∑ Wx + b ) = -w tanh( ∑ -Wx - b )
This function helps to define a unique representative on the equivalence class of neural networks that compute the same function.

Assumption: The activation function is odd, such as tanh.
"""
function normalize_neuron_signs(nn::Flux.Chain)::Flux.Chain
    for layer_idx in 1:length(nn.layers) - 1
        for neuron_idx in 1:size(Flux.params(nn.layers[layer_idx])[2])[1]
            nn = normalize_sign_in_neuron(nn, layer_idx, neuron_idx)
        end
    end
    return nn
end

"""
Do sign flip in a neuron if the bias is negative. Do nothing if the bias is positive.
    If the bias is zero, the sign of the first incoming weight is flipped such that it is positive. If this weight is also zero ...
"""
function normalize_sign_in_neuron(nn::Flux.Chain, layer_idx::Int, neuron_idx::Int; device = Flux.gpu)::Flux.Chain

    nn = nn |> Flux.cpu

    @assert layer_idx + 1 <= length(nn.layers) "No outgoing weights. No sign-flip symmetry in the last layer."

    b = Flux.params(nn.layers[layer_idx])[2][neuron_idx]
    incoming_W = Flux.params(nn.layers[layer_idx])[1][neuron_idx, :]
    outgoing_W = Flux.params(nn.layers[layer_idx + 1])[1][:, neuron_idx]

    if b > 0
        return nn
    elseif b < 0
        nn = do_sign_flip(nn, layer_idx, neuron_idx)
        return nn
    end

    # if bias is zero, flip the sign of the first incoming weight
    for w in incoming_W
        if w > 0
            return nn
        elseif w < 0
            nn = do_sign_flip(nn, layer_idx, neuron_idx)
            return nn
        end
    end

    for w in outgoing_W
        if w > 0
            return nn
        elseif w < 0
            nn = do_sign_flip(nn, layer_idx, neuron_idx)
            return nn
        end
    end

    # this condition can only be reached if all parameters are zero
    return nn |> device
end

"""
Flip the sign of all weights and bias connected to a neural network node.
"""
function do_sign_flip(nn::Flux.Chain, layer_idx::Int, neuron_idx::Int)::Flux.Chain
    b = Flux.params(nn.layers[layer_idx])[2][neuron_idx]
    incoming_W = Flux.params(nn.layers[layer_idx])[1][neuron_idx, :]
    outgoing_W = Flux.params(nn.layers[layer_idx + 1])[1][:, neuron_idx]

    Flux.params(nn.layers[layer_idx])[2][neuron_idx] = -b
    Flux.params(nn.layers[layer_idx])[1][neuron_idx, :] = -incoming_W
    Flux.params(nn.layers[layer_idx + 1])[1][:, neuron_idx] = -outgoing_W

    return nn
end