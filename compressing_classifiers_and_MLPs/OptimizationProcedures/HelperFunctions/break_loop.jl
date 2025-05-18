
using Statistics
using DiffEqFlux: collocate_data, EpanechnikovKernel
using Plots, Revise
plotlyjs()

"""
    is_saturated(mylog::Vector, smoothing_window::Int; plot_graph::Bool = false, min_possible_deviation=1e-10, return_plot::Bool=false, twosided=false)

    This function determines whether a given loss curve is saturated. It takes in the logged loss curve data mylog and uses the last smoothing_window+2 steps of this log. If length(mylog) < smoothing_window + 2, then it just returns false. Otherwise, it splits the last smoothing_window steps into two halves, determines the average of each and then checks if the 2nd average is bigger or equal to the first one, up to the variation in the data in those halves. 
    
    To determine the variation of the data, which itself follows a curve with varying mean value, we fit smoothed curves to the two halves, subtract the y-values of the smooth curves from the y-values of the data and then compute the standard deviation of the result. To compute the smoothed curves, we use the collocate_data function from the DiffEqFlux library. The collocate_data function employs an Epanechnikov kernel to determine the smoothed curve.

    The idea behind this slightly more involved saturation criterion is that this process can take into account trends of the loss curve, even when the loss curve itself might be noisy (and thus does not allow for a simpler procedure in which one simply triggers saturation / convergence as soon as the validation loss increases for the first time). Furthermore, it can be used both for validation loss as well as for train loss saturation / convergence.

    Returns a Boolean that indicates whether saturation took place or not.

    Arguments:
        - `mylog`: The data of the curve for which saturation should be checked. For example, this could be the array containing the validation or train loss.
        - `smoothing_window`: The number of epochs over which the smoothing and averaging should be performed. Bigger values are appropriate if one expects more noise in mylog.
        - `plot_graph`: This optional parameter can be set to true if one additionally wishes to see a plot of the smoothed curves and their mean values to visualize the saturation. It should be set to false during training.
        - `min_possible_deviation`: This value allows to trigger saturation, even when the 2nd mean value is (min_possible_deviation + variation) lower than the 1st mean value. Bigger values trigger saturation earlier.
        - `return_plot`: Setting this to `true` will return the saturation plot instead of the Boolean value.
        - `twosided`: If twosided is `true`, then the 2nd average does not only have to be higher or equal to the 1st average but both averages have to be within (min_possible_deviation + variation) of each other. This is a stronger saturation requirement and often too strong but might be useful in certain situations where the loss fluctuates on bigger time scales and one wants to trigger saturation only once those fluctuations eventually become smaller than (min_possible_deviation + variation).
"""
function is_saturated(mylog::Vector, smoothing_window::Int; plot_graph::Bool = false, min_possible_deviation=1e-10, return_plot::Bool=false, twosided=false)
    if length(mylog) < smoothing_window + 2
        return false
    else
        # divide the smoothing window in 2 parts:
        th = cld(smoothing_window,2) # cld(a,b) outputs the smallest integer bigger or equal to a/b
        
        # convert the mylogs in those two parts into matrices:
        w1 = reshape(mylog[end-2*th+1:end-th], 1, th)
        w2 = reshape(mylog[end-th+1:end], 1, th)
        
        # fit smooth curves to w1 and w2:
        _, u1 = collocate_data(w1, 1:th, EpanechnikovKernel())
        _, u2 = collocate_data(w2, 1:th, EpanechnikovKernel())
        # other kernels: https://docs.sciml.ai/DiffEqFlux/stable/utilities/Collocation/
        # example: https://docs.sciml.ai/DiffEqFlux/stable/examples/collocation/
        
        # compute mean and standard deviation:
        m1 = mean(w1)
        m2 = mean(w2)
        st1 = std(w1 .- u1)
        st2 = std(w2 .- u2)

        if plot_graph
            xs = collect(1:length(mylog))
            xss = xs[end-2*th+1:end]
            mx = [xss[argmin(abs.(u1 .- m1))[2]], xss[th + argmin(abs.(u2 .- m2))[2]]]
            p = Plots.plot(xs,mylog)
            p = Plots.plot(xss,mylog[end-2*th+1:end])
            p = Plots.scatter!(mx,[m1,m2])
            p = Plots.plot!(xss[end-2*th+1:end-th],[u1', (u1 .+ (st1 + min_possible_deviation))', (u1 .- (st1 + min_possible_deviation))'], legend=false)
            p = Plots.plot!(xss[end-th+1:end],[u2', (u2 .+ (st2 + min_possible_deviation))', (u2 .- (st2 + min_possible_deviation))'], legend=false)
            c1 = fill(m1-(st1 + min_possible_deviation),(2*th,1))
            c2 = fill(m2+(st2 + min_possible_deviation),(2*th,1))
            p = Plots.plot!(xss[end-2*th+1:end],c1,ls=:dash, legend=false)
            p = Plots.plot!(xss[end-2*th+1:end],c2,ls=:dash, legend=false)
        end

        # for batch processing: return the plot rather than displaying it
        if return_plot
            return p
        elseif plot_graph
            display(p)
        end

        # if abs(m1-m2) < st1 + st2
        # if m1 - st1 < m2 && m1 < m2 + st2
        if twosided
            criterion = abs(m1-m2)
        else
            criterion = m1-m2
        end
        if criterion <= min(st1, st2) + min_possible_deviation # for twosided, this corresponds to 
        # if m1 - st1 < m2 && m2 < m1 + st1 && m2 - st2 < m1 && m1 < m2 + st2
            return true
        else
            return false
        end
    end

end

function moving_average_full(data::(Vector{T} where T), window_size::Int)::(Vector{T} where T)
    n = length(data)
    moving_avg = zeros(eltype(data), length(data))
    
    for i in 1:n
        start_idx = max(1, i - window_size + 1)
        end_idx = min(n, i + window_size - 1)
        window = data[start_idx:end_idx]
        moving_avg[i] = mean(window)
    end
    
    return moving_avg
end

function smoothed_log(mylog::Vector, finetuning_start_epoch::Int; min_possible_deviation=1f-10, show_data=true, moving_avg_window_size=nothing)
    # convert the logs in those two parts into matrices:
    w1 = reshape(mylog[1:finetuning_start_epoch], 1, finetuning_start_epoch)
    w2 = reshape(mylog[finetuning_start_epoch+1:end], 1, length(mylog)-finetuning_start_epoch)
    
    if !isnothing(moving_avg_window_size) && isa(moving_avg_window_size, Int)
        u1 = moving_average_full(mylog[1:finetuning_start_epoch], moving_avg_window_size)
        u2 = moving_average_full(mylog[finetuning_start_epoch+1:end], moving_avg_window_size)
    else
        # fit smooth curves to w1 and w2:
        u1 = vec(collocate_data(w1, 1:finetuning_start_epoch, EpanechnikovKernel())[2])
        u2 = vec(collocate_data(w2, 1:length(mylog)-finetuning_start_epoch, EpanechnikovKernel())[2])
        # other kernels: https://docs.sciml.ai/DiffEqFlux/stable/utilities/Collocation/
        # example: https://docs.sciml.ai/DiffEqFlux/stable/examples/collocation/
    end
    
    # compute mean and standard deviation:
    m1 = mean(w1)
    m2 = mean(w2)
    st1 = std(w1 .- u1)
    st2 = std(w2 .- u2)

    xs1 = collect(1:finetuning_start_epoch)
    xs2 = collect(finetuning_start_epoch+1:length(mylog))
    xss = collect(1:length(mylog))
    mx = [xss[argmin(abs.(u1 .- m1))], xss[finetuning_start_epoch + argmin(abs.(u2 .- m2))]]
    p = 
    if show_data
        p = Plots.plot(xss,mylog)
        p = Plots.scatter!(mx,[m1,m2])
        p = Plots.plot!(xss[1:finetuning_start_epoch],[u1, (u1 .+ (st1 + min_possible_deviation)), (u1 .- (st1 + min_possible_deviation))], legend=false)
    else
        p = Plots.plot(xss[1:finetuning_start_epoch],[u1, (u1 .+ (st1 + min_possible_deviation)), (u1 .- (st1 + min_possible_deviation))], legend=false)
    end
    p = Plots.plot!(xss[finetuning_start_epoch+1:end],[u2, (u2 .+ (st2 + min_possible_deviation)), (u2 .- (st2 + min_possible_deviation))], legend=false)
    # c1 = fill(m1-(st1 + min_possible_deviation),(finetuning_start_epoch,1))
    # c2 = fill(m2+(st2 + min_possible_deviation),(length(mylog)-finetuning_start_epoch,1))
    # p = Plots.plot!(xss[1:finetuning_start_epoch],c1,ls=:dash, legend=false)
    # p = Plots.plot!(xss[finetuning_start_epoch+1:end],c2,ls=:dash, legend=false)
    display(p)
end

# function mask_from_params(ps,dtype)
#     ms = deepcopy(ps)
#     for (l1,l2) in zip(ms,ps)
#         for (w1,w2) in zip(l1,l2)
#             w1 .= w2 .== zero(dtype)
#         end
#     end
#     return ms
# end

# function check_equal(m1,m2)
#     for (l1,l2) in zip(m1,m2)
#         for (w1,w2) in zip(l1,l2)
#             if size(w1)!=size(w2) # first check if their dimensions agree
#                 return false
#             else
#                 if !all(w1.==w2) # only then check if entries agree
#                     return false
#                 end
#             end
#         end
#     end
#     return true
# end

# function copy_params!(m1,m2)
#     for (l1,l2) in zip(m1,m2)
#         for (w1,w2) in zip(l1,l2)
#             w1.=w2
#         end
#     end
# end
