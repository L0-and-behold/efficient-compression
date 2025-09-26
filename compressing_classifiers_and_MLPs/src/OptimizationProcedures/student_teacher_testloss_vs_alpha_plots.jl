using JLD2
using Plots
using Statistics
using Printf
plotlyjs()

alphas = Float32.([0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1,1.5,2,5,10])

@load "all_test_errs.jld2" all_test_errs
test_errs_matrix = hcat(all_test_errs...)

means = Statistics.mean(test_errs_matrix, dims=1)'
stds = Statistics.std(test_errs_matrix, dims=1)'

p1 = Plots.plot(alphas, means, ribbon = stds, label="", z_order=1, fillalpha=0.2,yscale=:log10)
p1 = Plots.scatter!(p1, alphas, means)

endalpha=11
p2 = Plots.plot(alphas[1:endalpha], means[1:endalpha], ribbon = stds, label="", z_order=1, fillalpha=0.2,yscale=:log10)
p2 = Plots.scatter!(p2, alphas[1:endalpha], means[1:endalpha])


medians = Statistics.median(test_errs_matrix, dims=1)'
q1s = mapslices(x -> quantile(x, 0.25), test_errs_matrix, dims=1)'
q3s = mapslices(x -> quantile(x, 0.75), test_errs_matrix, dims=1)'
lower_errors = medians .- q1s
upper_errors = q3s .- medians

p2 = Plots.plot(alphas, medians, ribbon = (lower_errors, upper_errors), label="", z_order=1, fillalpha=0.2, yscale=:log)

endalpha=11
ytickvals = [0.0075,0.00775,0.008,0.00825,0.0085,0.00875,0.09]
p2 = Plots.plot(alphas[1:endalpha], medians[1:endalpha], ribbon = (lower_errors, upper_errors), label="", z_order=1, fillalpha=0.4, legend=nothing, lw=3, xticks=[n*0.1 for n in 0:endalpha-1], yticks=(ytickvals,[string(x) for x in ytickvals]),
minorticks=true,
grid=true, tickfontsize=13)
p2 = Plots.scatter!(p2, alphas[1:endalpha], medians[1:endalpha])
10^(-2.10)