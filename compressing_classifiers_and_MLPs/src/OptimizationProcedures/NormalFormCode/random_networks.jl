"""
Neural network generation and manipulation utilities for teacher-student learning.

Provides functions to create random networks, embed smaller networks into larger ones,
and perform neuron shuffling while preserving network functionality.
"""

import Random, CUDA

include("NonDegenerateProjection.jl")
import .NonDegenerateProjection: project_onto_F

"""
Create a dense student network with random Glorot-initialized weights.

Generates a 3-layer feedforward network with tanh_fast activation for hidden layers
and linear activation for the output layer, suitable for student network training.

# Arguments
- `input_dim::Int`: Dimension of input layer
- `hidden_1::Int`: Number of neurons in first hidden layer
- `hidden_2::Int`: Number of neurons in second hidden layer  
- `output_dim::Int`: Dimension of output layer
- `rng`: Random number generator (default: Random.default_rng())
- `device`: Target device function (default: Flux.gpu)

# Returns
- `Flux.Chain`: Random student network with Glorot uniform initialization

# Architecture
- Layer 1: input_dim → hidden_1 (tanh_fast activation)
- Layer 2: hidden_1 → hidden_2 (tanh_fast activation)
- Layer 3: hidden_2 → output_dim (linear activation)
"""
function get_random_student(input_dim, hidden_1, hidden_2, output_dim; rng=Random.default_rng(), device = Flux.gpu)
    init = Flux.glorot_uniform(rng)
    init(dims...) = Flux.glorot_uniform(rng, dims...)

    nn = Flux.Chain(
        Flux.Dense(input_dim, hidden_1, tanh_fast, init=init),
        Flux.Dense(hidden_1, hidden_2, tanh_fast, init=init),
        Flux.Dense(hidden_2, output_dim, init=init)
    )
    return nn |> device
end

"""
Create a dense teacher network with optional Fefferman-Markel projection.

Generates a teacher network using the same architecture as student networks,
with optional projection onto the Fefferman-Markel non-degenerate set for
theoretical guarantees.

# Arguments
- `input_dim::Int`: Dimension of input layer
- `hidden_1::Int`: Number of neurons in first hidden layer
- `hidden_2::Int`: Number of neurons in second hidden layer
- `output_dim::Int`: Dimension of output layer
- `rng`: Random number generator (default: Random.default_rng())
- `device`: Target device function (default: Flux.gpu)
- `teacher_from_F::Bool`: Apply Fefferman-Markel projection (default: true)

# Returns
- `Flux.Chain`: Teacher network, optionally projected to satisfy non-degeneracy conditions

# Notes
- When teacher_from_F=true, applies project_onto_F for theoretical properties
- Small teachers can be embedded into larger networks using pad_small_into_big
- Useful for generating controlled teacher-student learning scenarios
"""
function get_random_teacher(input_dim, hidden_1, hidden_2, output_dim; rng=Random.default_rng(), device = Flux.gpu, teacher_from_F = true)
    nn = get_random_student(input_dim, hidden_1, hidden_2, output_dim; rng=rng, device = Flux.cpu)

    if teacher_from_F
        nn = project_onto_F(nn; device = Flux.cpu)
    end

    return nn |> device
end

"""
Embed a small teacher network into a larger network architecture.

Places the small teacher's parameters in the top-left corner of a larger network,
with remaining parameters set to zero. Preserves the teacher's functionality
while providing a larger parameter space.

# Arguments
- `small::Flux.Chain`: Small teacher network to embed
- `hidden_1::Int`: Size of first hidden layer in large network
- `hidden_2::Int`: Size of second hidden layer in large network
- `device`: Target device function (default: Flux.gpu)

# Returns
- `Flux.Chain`: Large network containing the embedded small teacher

# Process
1. Creates zero-initialized large network with same input/output dimensions
2. Copies small teacher's weights to top-left submatrices of large network
3. Copies small teacher's biases to first elements of large bias vectors

# Notes
- Input and output dimensions determined from small teacher
- Large network dimensions must be ≥ corresponding small network dimensions
- Resulting network computes same function as small teacher on embedded subspace
- Can be combined with shuffle() for neuron reordering
"""
function pad_small_into_big(small::Flux.Chain, hidden_1::Int, hidden_2::Int; device=Flux.gpu)::Flux.Chain

    small = small |> Flux.cpu

    input_dim = size(Flux.params(small.layers[1])[1])[2]
    output_dim = size(Flux.params(small.layers[3])[1])[1]

    big = get_random_teacher(input_dim, hidden_1, hidden_2, output_dim) |> Flux.cpu
    for p in Flux.params(big)
        p .= 0
    end

    for layer in 1:3
        # weights
        W = Flux.params(small.layers[layer])[1]
        s = size(W)
        Flux.params(big.layers[layer])[1][1:s[1], 1:s[2]] .= W 
        # biases
        b = Flux.params(small.layers[layer])[2]
        s = size(b)
        Flux.params(big.layers[layer])[2][1:s[1]] .= b
    end

    small = small |> device

    return big |> device
end

"""
Randomly shuffle neurons in hidden layers while preserving network function.

Performs random permutations of neurons in both hidden layers and adjusts
all connecting weights and biases accordingly to maintain the same computed function.

# Arguments
- `nn::Flux.Chain`: Neural network with exactly 3 layers (2 hidden)
- `rng`: Random number generator (default: Random.default_rng())
- `device`: Target device function (default: Flux.gpu)

# Returns
- `Flux.Chain`: Functionally equivalent network with shuffled hidden neurons

# Process
1. Generate random permutations for hidden layer neurons
2. Reorder weight matrix rows/columns according to permutations
3. Reorder bias vectors according to permutations
4. Input and output layers remain unchanged to preserve function

# Notes
- Network function remains identical after shuffling
- Only hidden layer neurons are permuted (not input/output)
- Useful for creating multiple representations of the same function
- Can reveal symmetries and redundancies in network structure
"""
function shuffle(nn::Flux.Chain; rng=Random.default_rng(), device=Flux.gpu)
    nn = nn |> Flux.cpu

    if length(nn.layers) != 3
        error("The network must have exactly 2 layers")
    end

    """
    Generate random permutations for hidden layer neurons.
    
    Returns permutation dictionaries p and inverse permutations q,
    where input (layer 0) and output (layer 3) remain unchanged.
    """
    function _random_neuron_permutations(nn::Flux.Chain; rng=Random.default_rng())::Tuple{Dict, Dict}
        
        if length(nn.layers) != 3
            error("The network must have exactly 2 layers")
        end
    
        ## permutations
        p = Dict()
        q = Dict() # inverse of p
    
        # layer 0 - input does not get permuted
        a = size(Flux.params(nn.layers[1])[1])[2]
        permutation_vector = [x for x in 1:a]
        inv_permutation_vector = invperm(permutation_vector)
        p[0] = permutation_vector
        q[0] = inv_permutation_vector
    
        # hidden layers
        for i in 1:2
            a = length(Flux.params(nn.layers[i])[2])
            permutation_vector = randperm(rng,a)
            inv_permutation_vector = invperm(permutation_vector)
            p[i] = permutation_vector
            q[i] = inv_permutation_vector
        end

        # layer 3 - output does not get permuted as this would change the computed function
        a = size(Flux.params(nn.layers[3])[1])[1]
        permutation_vector = [x for x in 1:a]
        inv_permutation_vector = invperm(permutation_vector)
        p[3] = permutation_vector
        q[3] = inv_permutation_vector
    
        return p, q
    end

    # p are the permutations of neurons, q are the inverse permutations
    # q[k][l] is the inverse permutation of the l-th neuron in layer k, 
    p, q = _random_neuron_permutations(nn; rng=rng) 

    new = adjust_params(nn, p, q; device=device)

    nn = nn |> device
    return new
end

"""
Apply neuron permutations to network weights and biases.

Helper function that adjusts all network parameters according to given 
permutation dictionaries, ensuring the network function remains unchanged.

# Arguments
- `nn::Flux.Chain`: Neural network with exactly 3 layers
- `p::Dict`: Permutation vectors for each layer
- `q::Dict`: Inverse permutation vectors for each layer
- `device`: Target device function (default: Flux.gpu)

# Returns
- `Flux.Chain`: Network with permuted parameters

# Process
- For each layer, permutes incoming weights according to previous layer permutation
- Permutes outgoing weights according to current layer permutation  
- Permutes bias vectors according to current layer permutation

# Notes
- p[i] contains permutation for layer i neurons
- q[i] contains inverse permutation for layer i
- Input (layer 0) and output (layer 3) permutations are identity
- Handles edge cases with empty weight matrices
"""
function adjust_params(nn::Flux.Chain, p::Dict, q::Dict; device=Flux.gpu)::Flux.Chain

    if length(nn.layers) != 3
        error("The network must have exactly 2 layers")
    end
    for i in 0:3
        if !isa(p[i], Vector) || !isa(q[i], Vector)
            error("p, q must be dictionaries of permutation vectors, one per layer.")
        end
        try
            invperm(p[i])
        catch
            error("p[$i] must be a permutation vector.")
        end
    end

    new = deepcopy(nn)

    # adjust weights
    for i in 1:3 # 3 layers
        W = Flux.params(new.layers[i])[1]
        ω = deepcopy(W)

        # respect permutation of neurons in previous layer
        ω = [ω[:, i] for i in 1:size(ω, 2)]
        ω = ω[p[i-1]]
        ω = [x for x in hcat(ω...)]

        # do permutation of neurons in this layer
        ω = [ω[i, :] for i in 1:size(ω, 1)]
        
        try
            ω = ω[p[i]]
        catch e
            isa(e, BoundsError) && ω == Any[] && continue # do nothing if weights are completely empty
            rethrow(e)
        end

        ω = [x for x in hcat(ω...)']


        try
            Flux.params(new.layers[i])[1] .= ω
        catch e
            isa(e, DimensionMismatch) && (size(ω)[1] == 0 || size(ω)[2] == 0) && continue # do nothing if weights are completely empty
            rethrow(e)
        end
    end

    # adjust biases
    for i in 1:3 # 3 layers
        b = Flux.params(new.layers[i])[2]
        β = deepcopy(b)
        Flux.params(new.layers[i])[2] .= β[p[i]]
    end

    return new |> device
end

