"""
NonDegenerateProjection module for Fefferman-Markel network conditions.

Provides functions to check and enforce the "generic" conditions for fully-connected 
feedforward networks with tanh activation functions as defined by Fefferman-Markel theory.

# Fefferman-Markel Generic Conditions
    https://proceedings.neurips.cc/paper_files/paper/1993/file/e49b8b4053df9505e1f48c3a701c0682-Paper.pdf

For a network to be non-degenerate, it must satisfy:

1. **Non-zero parameters**: ∀ b_j^l ∈ θ: b_j^l ≠ 0 ∧ ∀ w_jk^l ∈ θ: w_jk^l ≠ 0
2. **Unique absolute values**: |b_j^l| ≠ |b_j'^l| ∀ l, j, j' ≠ j  
3. **Rational weight ratios**: w_jk^l/w_j'k^l ≠ p/q for p,q ∈ ℕ: 1 ≤ max(p,q) ≤ D_l²
   where D_l is the output dimension of layer l

# Main Functions

## Network Projection
- `project_onto_F`: Project entire network to satisfy all conditions

## Condition Enforcement  
- `ensure_non_zero`: Eliminate zero parameters (condition 1)
- `ensure_non_equality`: Ensure unique absolute values (condition 2)
- `rationality_projection`: Enforce rational ratio constraints (condition 3)

## Validation
- `rationality_condition_is_met`: Check if two values satisfy rationality condition

## Utilities
- `pertubation`: Generate small perturbations preserving floating-point precision
- `float_to_exact_rational`: Robust float-to-rational conversion

# Usage Example

```julia
# Project network to satisfy Fefferman-Markel conditions
nn_projected = project_onto_F(nn; device=gpu)
```

# Notes
- All perturbations preserve numerical precision and are minimal
- Projections maintain network functionality while satisfying theoretical requirements
- Functions handle both CPU and GPU tensors automatically
"""

module NonDegenerateProjection

using Flux
using Flux: Chain, params, cpu, gpu

export rationality_condition_is_met, ensure_non_equality, ensure_non_zero, pertubation, project_onto_F, rationality_projection

"""
Project a neural network to satisfy all Fefferman-Markel generic conditions.

Applies minimal perturbations to ensure the network satisfies non-degeneracy conditions
while preserving proximity to the original parameterization in ℝⁿ standard norm.

# Arguments
- `nn::Union{Chain, Flux.Chain}`: Input neural network
- `device`: Target device function (default: gpu)

# Returns
- `Union{Chain, Flux.Chain}`: Projected network satisfying all conditions

# Process
1. Eliminates zero parameters
2. Ensures unique absolute values for biases
3. Enforces rationality constraints on weight ratios
"""
function project_onto_F(nn::Union{Chain, Flux.Chain}; device = gpu)::Union{Chain, Flux.Chain}
    nn = cpu(nn)

    nn = ensure_non_zero(nn)
    nn = ensure_non_equality(nn; device = cpu)
    nn = rationality_projection(nn; device = cpu)

    return device(nn)
end

"""
Eliminate all zero parameters in the network (Condition 1).

# Arguments
- `nn::Union{Chain, Flux.Chain}`: Neural network to modify

# Returns
- Modified network with no zero parameters
"""
function ensure_non_zero(nn::Union{Chain, Flux.Chain}) 
    for p in params(nn)
        p .= pertubation.(p)
    end
    
    return nn
end

"""
Generate a small perturbation of a floating-point number.

Preserves floating-point precision while ensuring the perturbed value differs 
from the original. Uses machine epsilon scaling for minimal perturbations.

# Arguments
- `x::T`: Value to perturb
- `temperature`: Perturbation magnitude scaling factor (default: 10)

# Returns
- `T`: Perturbed value guaranteed to differ from input

# Notes
- For zero inputs, adds random signed epsilon-scaled noise
- For non-zero inputs, adds proportional epsilon-scaled noise
- Increases temperature on failed attempts (up to 10 tries)
"""
function pertubation(x::T; temperature = 10) where T <: AbstractFloat
    if x == 0
        return x + temperature * rand() * eps(T) * rand([-1, 1])
    end

    for _ in 1:10
        x_pertubed = x + temperature * x * rand() * eps(T)
        temperature += 1
        x_pertubed != x && return x_pertubed
    end
    error("Pertubation failed. Not able to find a pertubation of $x after 10 attempts.")
end

"""
Generate a small perturbation of a rational number, converted to specified float type.

# Arguments
- `x::Rational{T}`: Rational number to perturb
- `type::DataType`: Target floating-point type (Float16, Float32, or Float64)
- `temperature`: Perturbation magnitude scaling factor (default: 10)

# Returns
- `Rational`: Perturbed rational number converted to float precision

# Notes
- Handles high-precision rational perturbations with float conversion
- Uses fallback rational conversion for numerical stability
"""
function pertubation(x::Rational{T}, type::DataType; temperature = 10) where T <: Union{Int64, BigInt}
    @assert type == Float64 || type == Float32 || type == Float16 "Error: type must be an AbstractFloat"

    for _ in 1:10
        x_pertubed = try
            x + Rational(temperature * rand() * eps(type)) * x * rand([-1, 1])
        catch
            x + float_to_exact_rational(temperature * rand() * eps(type)) * x * rand([-1, 1])
        end

        x_pertubed = try
            Rational(type(x_pertubed)) # rationals can represent pertubations to much greater precision than floats. So we need to convert it back to a float to avoid that the pertubation is lost in floating point precision.
        catch
            float_to_exact_rational(type(x_pertubed)) # rationals can represent pertubations to much greater precision than floats. So we need to convert it back to a float to avoid that the pertubation is lost in floating point precision.
        end

        temperature += 1
        x_pertubed != x && return x_pertubed
    end
    error("Pertubation failed. Not able to find a pertubation of $x after 10 attempts.")
end

"""
Ensure a value differs from all values in a vector.

# Arguments
- `bias::T`: Value to make unique
- `other_biases::Vector{T}`: Vector of values to differ from
- `temperature`: Initial perturbation magnitude (default: 10)

# Returns
- `T`: Perturbed value guaranteed to differ from all values in other_biases
"""
function ensure_non_equality(bias::T, other_biases::Vector{T}; temperature = 10) where T <: AbstractFloat
    for _ in 1:100
        if any(x -> x == bias, other_biases)
            bias = pertubation(bias; temperature)
            temperature += 1
        else
            return bias
        end
    end
    error("Not able to ensure that the bias $x is unique after 100 attempts.")
end


"""
Ensure all values in a vector have unique absolute values.

# Arguments
- `biases::Vector{T}`: Vector of values to make unique
- `temperature`: Initial perturbation magnitude (default: 10)

# Returns
- `Vector{T}`: Vector with all unique absolute values
"""
function ensure_non_equality(biases::Vector{T}; temperature = 10) where T <: AbstractFloat
    result = similar(biases)
    for i in eachindex(biases)

        mask = trues(length(biases))
        mask[i] = false
        
        result[i] = ensure_non_equality(biases[i], biases[mask]; temperature = temperature)
    end
    return result
end

"""
Enforce unique absolute bias values across all layers (Condition 2).

# Arguments
- `nn::Union{Chain, Flux.Chain}`: Neural network to modify
- `device`: Target device function (default: gpu)
- `temperature`: Initial perturbation magnitude (default: 10)

# Returns
- Modified network with unique absolute bias values in each layer
"""
function ensure_non_equality(nn::Union{Chain, Flux.Chain}; device = gpu, temperature = 10)
    nn = cpu(nn)

    function _f(layer_index::Int, biases::Vector{T}) where T <: AbstractFloat
        biases = ensure_non_equality(biases; temperature = temperature)
        params(nn.layers[layer_index])[2] .= biases
    end

    # the biases are stored in params(nn.layers[i])[2]
    map(_f, 1:length(nn.layers), [params(nn.layers[i])[2] for i in 1:length(nn.layers)])

    return device(nn)
end

"""
Check if two rational numbers satisfy the rationality condition (Condition 3).

# Arguments
- `a::Rational`: First rational number
- `b::Rational`: Second rational number  
- `D_l::Number`: Output dimension of the layer

# Returns
- `Bool`: True if |a/b| ≠ p/q for natural numbers p,q ≤ D_l²

# Notes
- Uses smallness bound of 100 × D_l² for practical computation
- Handles overflow with BigInt arithmetic when necessary
"""
function rationality_condition_is_met(a::Rational, b::Rational, D_l::Number)::Bool
    
    smallness_bound = 100 * D_l^2
    
    (a == 0 || b == 0 || a == b) && return false

    a, b = abs(a), abs(b)

    try
        ratio = a // b
        return !(ratio.num <= smallness_bound || ratio.den <= smallness_bound)
    catch OverflowError
        big_a, big_b = big(a), big(b)
        ratio = big_a // big_b
        return !(ratio.num <= smallness_bound || ratio.den <= smallness_bound)
    end
end


"""
Apply rationality projection to a vector of weights.

Perturbs each weight until it satisfies the rationality condition with all 
other weights in the vector.

# Arguments
- `weight_vector::Vector{T}`: Vector of weights to project
- `D_l::Number`: Output dimension of the layer
- `temperature`: Initial perturbation magnitude (default: 10)

# Returns
- `Vector{T}`: Projected weight vector satisfying rationality conditions
"""
function rationality_projection(weight_vector::Vector{T}, D_l::Number; temperature = 10)::Vector{T} where T <: AbstractFloat
    weights_rational = try
        Rational.(weight_vector)
    catch
        float_to_exact_rational.(weight_vector)
    end
    
    result = similar(weights_rational)
    # other_weights = similar(weights_rational, length(weights_rational) - 1)
    
    @inbounds for i in eachindex(weights_rational)
        weight = weights_rational[i]
        # other_weights[1:i-1] .= view(weights_rational, 1:i-1)
        # other_weights[i:end] .= view(weights_rational, i+1:length(weights_rational))
        mask = trues(length(weights_rational))
        mask[i] = false

        attempts = 0
        # condition_met_with = [rationality_condition_is_met(weight, ow, D_l) for ow in other_weights]
        condition_met_with = [rationality_condition_is_met(weight, ow, D_l) for ow in weights_rational[mask]]
        
        while !all(condition_met_with)
            weight = pertubation(weight, T; temperature = temperature)
            temperature += 1
            attempts += 1
            
            if attempts > 50
                error("Not able to find a pertubation of $weight after 50 attempts.")
            end
            
            # condition_met_with .= [rationality_condition_is_met(weight, ow, D_l) for ow in other_weights]
            condition_met_with .= [rationality_condition_is_met(weight, ow, D_l) for ow in weights_rational[mask]]
        end
        
        result[i] = weight
    end
    
    return T.(result)
end

"""
Apply rationality projection to all weight matrices in a neural network.

Projects column vectors (outgoing weights per neuron) to satisfy rationality 
conditions based on the output layer dimension.

# Arguments
- `nn::Union{Flux.Chain, Chain}`: Neural network to project
- `temperature`: Initial perturbation magnitude (default: 10)
- `device`: Target device function (default: gpu)

# Returns
- `Union{Flux.Chain, Chain}`: Network with rationality-projected weights
"""
function rationality_projection(nn::Union{Flux.Chain, Chain}; temperature = 10, device = gpu)::Union{Flux.Chain, Chain}

    nn = cpu(nn)
    D_l = length(params(nn.layers[end])[2])
    matrix_T = typeof(params(nn.layers[1])[1])
    
    function _project_single_layer(layer::Flux.Dense)
        W = params(layer)[1]

        # skip empty layers
        if 0 in size(W)
            return
        end
        
        float_T = typeof(W[1])

        incoming_weights = rationality_projection.([Vector{float_T}(W[:, i]) for i in 1:size(W, 2)], D_l; temperature = temperature)
        
        W = matrix_T(hcat(incoming_weights...))
        
        params(layer)[1] .= W
    end

    map(_project_single_layer, nn.layers)

    return device(nn)
end

"""
Robust conversion from floating-point to exact rational representation.

Fallback function for `Rational()` constructor that handles edge cases and 
provides more robust conversion, especially for Float16 values.

# Arguments
- `x::T`: Floating-point number to convert

# Returns
- `Rational{BigInt}`: Exact rational representation

# Notes
- Handles significand and exponent separately for precision
- Uses BigInt arithmetic to avoid overflow
- Slower than default Rational() but more robust
"""
function float_to_exact_rational(x::T)::Rational{BigInt} where T <: Union{Float16, Float32, Float64} 

    try
        return Rational(x)
    catch
    end

    if isinteger(x)
        return Rational{BigInt}(BigInt(x), 1)
    end
    
    s = sign(x) # 1 or -1
    e = exponent(x)
    f = significand(x)
    
    # Convert significand to rational
    num = BigInt(abs(trunc(Int64, f * BigInt(2)^precision(T))))
    den = BigInt(2)^precision(T)
    
    # Apply exponent
    if e >= 0
        num *= BigInt(2)^e
    else
        den *= BigInt(2)^(-e)
    end
    
    # Apply sign
    num *= BigInt(s)
    
    return Rational{BigInt}(num, den)
end

end # module