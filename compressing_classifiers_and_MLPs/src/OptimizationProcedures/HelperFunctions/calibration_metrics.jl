"""
    Calibration metrics for evaluating model confidence calibration.
    
    Implements Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    and reliability diagram data following Guo et al. 2017 "On Calibration of Modern Neural Networks".
"""

using NNlib: softmax

"""
    collect_logits_and_labels(tstate, dataset) -> (logits_matrix, integer_labels)

Run forward pass over all batches in `dataset`, returning concatenated logits and integer labels on CPU.
Handles both masked models (tstate.parameters.p) and unmasked models (tstate.parameters).

Returns:
- `logits`: Matrix of shape (num_classes, N) on CPU
- `labels`: Vector of integer labels (0-indexed) of length N on CPU
"""
function collect_logits_and_labels(tstate, dataset)
    cpu = Lux.cpu_device()
    st = testmode_states(tstate)
    inner_st = haskey(st, :st) ? st.st : st
    ps = haskey(tstate.parameters, :p) ? tstate.parameters.p : tstate.parameters

    all_logits = []
    all_labels = []

    for (x, y) in dataset
        logits_batch, _ = tstate.model(x, ps, inner_st)
        logits_cpu = cpu(logits_batch)  # (num_classes, batch_size)
        labels_cpu = cpu(y)             # one-hot (num_classes, batch_size)

        push!(all_logits, logits_cpu)
        push!(all_labels, labels_cpu)
    end

    logits = hcat(all_logits...)           # (num_classes, N)
    labels_onehot = hcat(all_labels...)    # (num_classes, N)
    # Convert one-hot to integer labels (0-indexed to match CIFAR convention)
    labels = vec(map(i -> argmax(labels_onehot[:, i]) - 1, 1:size(labels_onehot, 2)))

    return logits, labels
end

"""
    reliability_diagram_data(logits, labels; n_bins=15)

Compute per-bin statistics for a reliability diagram.

Arguments:
- `logits`: Matrix (num_classes, N) of raw model outputs (pre-softmax)
- `labels`: Vector of integer labels (0-indexed)
- `n_bins`: Number of equal-width confidence bins (default 15)

Returns a NamedTuple with:
- `bin_accuracies`: accuracy within each bin (NaN for empty bins)
- `bin_confidences`: mean confidence within each bin (NaN for empty bins)
- `bin_counts`: number of samples in each bin
- `bin_edges`: bin edge positions (length n_bins+1)
- `ece`: Expected Calibration Error
- `mce`: Maximum Calibration Error
"""
function reliability_diagram_data(logits::AbstractMatrix, labels::AbstractVector; n_bins::Int=15)
    probs = softmax(logits; dims=1)  # (num_classes, N)
    N = size(probs, 2)

    # Per-sample confidence and predicted class
    confidences = vec(maximum(probs; dims=1))                    # length N
    predictions = vec(map(i -> argmax(probs[:, i]) - 1, 1:N))   # 0-indexed

    # Correctness
    correct = predictions .== labels  # BitVector

    # Bin edges
    bin_edges = range(0.0f0, 1.0f0, length=n_bins + 1)

    bin_accuracies = fill(NaN32, n_bins)
    bin_confidences = fill(NaN32, n_bins)
    bin_counts = zeros(Int, n_bins)

    for b in 1:n_bins
        lo = bin_edges[b]
        hi = bin_edges[b + 1]
        # Last bin is inclusive on both sides
        if b == n_bins
            in_bin = (confidences .>= lo) .& (confidences .<= hi)
        else
            in_bin = (confidences .>= lo) .& (confidences .< hi)
        end
        count = sum(in_bin)
        bin_counts[b] = count
        if count > 0
            bin_accuracies[b] = sum(correct[in_bin]) / count
            bin_confidences[b] = sum(confidences[in_bin]) / count
        end
    end

    # ECE and MCE (only over non-empty bins)
    ece = 0.0f0
    mce = 0.0f0
    for b in 1:n_bins
        if bin_counts[b] > 0
            gap = abs(bin_accuracies[b] - bin_confidences[b])
            ece += (bin_counts[b] / N) * gap
            mce = max(mce, gap)
        end
    end

    return (
        bin_accuracies = bin_accuracies,
        bin_confidences = bin_confidences,
        bin_counts = bin_counts,
        bin_edges = collect(bin_edges),
        ece = ece,
        mce = mce,
    )
end

"""
    expected_calibration_error(logits, labels; n_bins=15) -> (ece, mce)

Compute Expected Calibration Error and Maximum Calibration Error.

Arguments:
- `logits`: Matrix (num_classes, N) of raw model outputs (pre-softmax)
- `labels`: Vector of integer labels (0-indexed)
- `n_bins`: Number of equal-width confidence bins (default 15)
"""
function expected_calibration_error(logits::AbstractMatrix, labels::AbstractVector; n_bins::Int=15)
    rd = reliability_diagram_data(logits, labels; n_bins=n_bins)
    return rd.ece, rd.mce
end
