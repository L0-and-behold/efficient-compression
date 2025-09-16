using Lux: cpu_device
using Plots, ColorSchemes, PlotlyJS, Revise


"""
    plot_weights(tstate; legend = :topright, title = "")

    This function plots the weights and biases of a multi-layer-perceptron with arbitrary layer number and sizes. The colors indicate the sign of the weights (blue for positive weights, red for negative weights) and the strength of the lines indicates the relative strength of the weights in the network.

    Arguments:
        - `tstate`: The Lux TrainState (encoding neural network model parameters and architecture).
        - `legend`: The position of the legend (default is `:topright`).
        - `title`: You guessed it, the plot title.
"""
function plot_weights(tstate; legend = :topright, title = "")

    if haskey(tstate.parameters,:p)
        tps = deepcopy(tstate.parameters.p) |> Lux.cpu_device()
    elseif haskey(tstate.parameters,:w)
        tps = deepcopy(tstate.parameters.w) |> Lux.cpu_device()
    else
        tps = deepcopy(tstate.parameters) |> Lux.cpu_device()
    end

    if tstate.model.name == "ascent model"
        layer_sizes = vcat([tstate.model.layers.layer_1.layer.in_dims],[tstate.model.layers[layer_key].layer.out_dims for layer_key in keys(tstate.model.layers)])
    else
        layer_sizes = vcat([tstate.model.layers.layer_1.in_dims],[tstate.model.layers[layer_key].out_dims for layer_key in keys(tstate.model.layers)])
    end
    max_layer_size = maximum(layer_sizes)

    positions = [[(i-1, j) for j in 1:size] for (i, size) in enumerate(layer_sizes)]
    
    # Function to map weight to color
    function weight_to_color(weight)
        if weight == 0.0
            return RGB(1, 1, 1)  # white
        elseif weight > 0
            return RGB(0, 0, 1)  # Blue for positive weights
        else
            return RGB(1, 0, 0)  # Red for negative weights
        end
    end

    # Function to map bias to color
    function bias_to_color(bias)
        colorscheme = ColorSchemes.RdYlGn
        normalized_bias = (bias + 1) / 2  # Assuming bias is in [-1, 1] range
        return get(colorscheme, normalized_bias)
    end

    p = Plots.plot(size=(600, 500), legend=legend, title=title)


    scaling_factor = max(20/max_layer_size, 1)
    linewdth(w) = max(abs(w) * scaling_factor, 0.4)

    # Draw connections with weights
    layer_number = 0
    for layer in tps
        layer_number += 1
        weights = layer.weight

        for i in 1:size(weights, 2)
            for j in 1:size(weights, 1)
                weight = weights[j, i]
                if weight == 0.0 # for performance reasons, we explicitly skip zero weights
                    continue
                end
                start, stop = positions[layer_number][i], positions[layer_number+1][j]
                Plots.plot!(p, [start[1], stop[1]], [start[2], stop[2]],
                    color=weight_to_color(weight),
                    alpha=1.0,
                    linewidth=linewdth(weight),
                    label="")
            end
        end
    end
    biases = [tps[layer_key].bias for layer_key in keys(tps)]
    # Draw nodes with colors based on bias
    for (i, pos) in enumerate(positions)
        color = i == 1 ? fill(:gray, length(pos)) : [bias_to_color(b) for b in biases[i-1]]
        scatter!(p, first.(pos), last.(pos),
                    markersize=4*scaling_factor,
                    color=color,
                    label=i == 1 ? "No bias (input)" : false)
    end

    # Add legend items for weight colors
    Plots.plot!(p, [], [], color=:blue, linewidth=2, label="Positive weight")
    Plots.plot!(p, [], [], color=:red, linewidth=2, label="Negative weight")

    # Add legend items for bias colors
    Plots.scatter!(p, [], [], color=bias_to_color(-0.8), markersize=6, label="Negative bias")
    Plots.scatter!(p, [], [], color=bias_to_color(0), markersize=6, label="Zero bias")
    Plots.scatter!(p, [], [], color=bias_to_color(0.8), markersize=6, label="Positive bias")

    Plots.xlims!(p, -0.5, 3.5)
    Plots.ylims!(p, 0.5, max_layer_size + 0.5)
    
    # Remove axis and ticks
    Plots.plot!(p, xticks=false, yticks=false, xaxis=false, yaxis=false)

    return p
end


"""
    net_scatter_plot(dataset_dev; aspect="cube", display_plot=true, scatterplot_col=nothing, combine_with_the_plot=nothing, name="dataset", show_legend=true)

    This function plots a scatter plot of a given dataset, as long as it has the correct input and output dimensions. The sum of in- and output dimension must not exceed 3.

    Arguments:
        - `dataset_dev`: The dataset to be plotted.
        - `aspect`: The aspect ratio setting. Default is `cube`.
        - `display_plot`: If false, then plot is returned but not displayed.
        - `scatterplot_col`: The color of the scatter points. Default depends on the plotted dimensions.
        - `combine_with_the_plot`: This option allows to combine the new plot with a previous plot object.
        - `name`: The name of the scatter points in the plot legend.
        - `show_legend`: Shows the legend, if true.
"""
function net_scatter_plot(dataset_dev; aspect="cube", display_plot=true, scatterplot_col=nothing, combine_with_the_plot=nothing, name="dataset", show_legend=true, camera=(1.5,-1.5,1.5), savepath=nothing,markersizeattribute=2, showticklabels=false)

    dataset = deepcopy(dataset_dev) |> Lux.cpu_device()

    function combine(combine_with_the_plot::Array, shape)
        return [combine_with_the_plot...,shape]
    end
    function combine(combine_with_the_plot,shape)
        if isnothing(combine_with_the_plot)
            return shape
        else
            return [combine_with_the_plot,shape]
        end
    end
    s1 = size(dataset[1])[1]
    s2 = size(dataset[2])[1]
    if s1 == 1 && s2 == 1
        if isnothing(scatterplot_col)
            scatterplot_col = "red"
        end
        scatter_plot = PlotlyJS.scatter(x=dataset[1][1,:], y=dataset[2][1,:], mode="markers", marker=attr(size=markersizeattribute, color=scatterplot_col), name = name,
        showlegend = show_legend)

        scatter_plot = combine(combine_with_the_plot,scatter_plot)

        p = PlotlyJS.plot(scatter_plot)
        if display_plot
            display(p)
        end
        if !isnothing(savepath)
            PlotlyJS.savefig(p, savepath * ".png", width=2560, height=1440)
        end
        return scatter_plot
    elseif s1 == 1 && s2 == 2
        in_out_dims = [(1,1),(2,1),(2,2)]
        if isnothing(scatterplot_col)
            scatterplot_col = "red"
        end
    elseif s1 == 2 && s2 == 1
        in_out_dims = [(1,1),(1,2),(2,1)]
        if isnothing(scatterplot_col)
            scatterplot_col = "blue"
        end
    else
        print("Dimensions are not appropriate for plotting.")
        return
    end
    d1 = dataset[in_out_dims[1][1]][in_out_dims[1][2], :]
    d2 = dataset[in_out_dims[2][1]][in_out_dims[2][2], :]
    d3 = dataset[in_out_dims[3][1]][in_out_dims[3][2], :]
    if aspect!="cube"
        x_range = maximum(d1) - minimum(d1)
        y_range = maximum(d2) - minimum(d2)
        z_range = maximum(d3) - minimum(d3)
    end
    scatter_plot = PlotlyJS.scatter3d(
        x = d1,
        y = d2,
        z = d3,
        mode = "markers",
        marker = attr(size=markersizeattribute, color = scatterplot_col),
        name = name,
        showlegend = show_legend
    )
    scatter_plot = combine(combine_with_the_plot,scatter_plot)
    if aspect=="cube"
        layout = Layout(
            title = "3D Scatter Plot",
            scene = Dict(
                :xaxis => attr(title="X1", showticklabels=showticklabels),
                :yaxis => attr(title="X2", showticklabels=showticklabels),
                :zaxis => attr(title="Y", showticklabels=showticklabels),
                :aspectratio => attr(x=1, y=1, z=1),
                # :aspectratio => attr(x=1/x_range, y=1/y_range, z=1/z_range)
                :camera => attr(
                    eye = attr(x=camera[1], y=camera[2], z=camera[3]),    # Camera position
                    center = attr(x=0, y=0, z=0),        # Look at point
                    up = attr(x=0, y=0, z=1)             # Up direction
                ),
            ),
            showlegend = show_legend
        )
    else
        layout = Layout(
            title = "3D Scatter Plot",
            scene = Dict(
                :xaxis => attr(title="X1", showticklabels=showticklabels),
                :yaxis => attr(title="X2", showticklabels=showticklabels),
                :zaxis => attr(title="Y", showticklabels=showticklabels),
                :aspectratio => attr(x=1/x_range, y=1/y_range, z=1/z_range),
                :camera => attr(
                    eye = attr(x=camera[1], y=camera[2], z=camera[3]),    # Camera position
                    center = attr(x=0, y=0, z=0),        # Look at point
                    up = attr(x=0, y=0, z=1)             # Up direction
                ),
            ),
            showlegend = show_legend
        )
    end
    p = PlotlyJS.plot(scatter_plot, layout)
    if display_plot
        display(p)
    end
    if !isnothing(savepath)
        PlotlyJS.savefig(p, savepath * ".png", width=2560, height=1440)
    end
    return scatter_plot
end


"""
    net_smooth_plot(tstate; N=100, display_plot=true, color_scheme="student", opacity=1.0, combine_with_the_plot=nothing, title=nothing)

    This function plots a smooth surface that corresponds to the function computed by some neural network, as long as the neural network has the correct input and output dimensions. The sum of in- and output dimension must not exceed 3.

    Arguments:
        - `tstate`: A Lux TrainState encoding neural model and parameters
        - `N`: The resolution of the plot grid.
        - `display_plot`: If false, then plot is returned but not displayed.
        - `color_scheme`: One can choose among 2 color schemes: 1) "student" and 2) "teacher".
        - `opacity`: The opacity of the plot. Useful for combined plots.
        - `combine_with_the_plot`: This option allows to combine the new plot with a previous plot object.
        - `title`: You guessed it, the plot title.
"""
function net_smooth_plot(tstate; N=100, display_plot=true, color_scheme="student", opacity=1.0, combine_with_the_plot=nothing, title=nothing, camera=(1.5,-1.5,1.5), savepath=nothing, showticklabels=false)

    if haskey(tstate.parameters,:p)
        tps = deepcopy(tstate.parameters.p) |> Lux.cpu_device()
    else
        tps = deepcopy(tstate.parameters) |> Lux.cpu_device()
    end
    if haskey(tstate.states,:st)
        tss = deepcopy(tstate.states.st) |> Lux.cpu_device()
    else
        tss = deepcopy(tstate.states) |> Lux.cpu_device()
    end
    dtype = typeof(tps.layer_1.bias[1])

    function combine(combine_with_the_plot::Array, shape)
        return [combine_with_the_plot...,shape]
    end
    function combine(combine_with_the_plot,shape)
        if isnothing(combine_with_the_plot)
            return shape
        else
            return [combine_with_the_plot,shape]
        end
    end

    if tstate.model[1].in_dims == 1
        x = range(-1, stop=1, length=N)
        x = dtype.(collect(x))
        if tstate.model[end].out_dims == 1
            if color_scheme == "teacher"
                line_color = "red"
                if isnothing(title)
                    title="Teacher"
                end
            else
                line_color = "blue"
                if isnothing(title)
                    title="Student"
                end
            end
            y = tstate.model(reshape(x,(1,N)), tps, tss)[1][1,:]
            l = PlotlyJS.scatter(
                    x = x,
                    y = y,
                    mode = "lines",
                    line = attr(color = line_color),
                    name = title
                )
            l = combine(combine_with_the_plot,l)
            layout = PlotlyJS.Layout(
                title = "Model(x)",
                xaxis = attr(title = "X", showticklabels=showticklabels),
                yaxis = attr(title = "Y", showticklabels=showticklabels),
                zaxis = attr(title = "Z", showticklabels=showticklabels),
                showlegend = true,
                scene=attr(
                    camera=attr(
                        eye=attr(x=camera[1], y=camera[2], z=camera[3]),
                        center=attr(x=0, y=0, z=0),
                        up=attr(x=0, y=0, z=1)
                    )
                )
            )
            p = PlotlyJS.plot(l,layout)
            if display_plot
                display(p)
            end
            if !isnothing(savepath)
                PlotlyJS.savefig(p, savepath * ".png", width=2560, height=1440)
            end
            return l
        elseif tstate.model[end].out_dims == 2

            yz = tstate.model(reshape(x,(1,N)), tps, tss)[1]
            y = yz[1,:]
            z = yz[2,:]

            if color_scheme == "teacher"
                line_color = "red"
                if isnothing(title)
                    title="Teacher"
                end
            else
                line_color = "blue"
                if isnothing(title)
                    title="Student"
                end
            end
            line_plot = PlotlyJS.scatter3d(
                x = x,
                y = y,
                z = z,
                mode = "lines",
                line = attr(width = 3, color = line_color, opacity=opacity, title=title),
                name = title
            )
            line_plot = combine(combine_with_the_plot,line_plot)
            layout = PlotlyJS.Layout(
                title = "Parametric plot of model(x)",
                scene = Dict(
                    :xaxis => attr(title = "X", showticklabels=showticklabels),
                    :yaxis => attr(title = "Y", showticklabels=showticklabels),
                    :zaxis => attr(title = "Z", showticklabels=showticklabels),
                    :aspectmode => "cube",
                    :camera => attr(
                        eye = attr(x=camera[1], y=camera[2], z=camera[3]),    # Camera position
                        center = attr(x=0, y=0, z=0),        # Look at point
                        up = attr(x=0, y=0, z=1)             # Up direction
                    ),
                ),
                showlegend = true
            )
            p = PlotlyJS.plot(line_plot, layout)
            if display_plot
                display(p)
            end
            if !isnothing(savepath)
                PlotlyJS.savefig(p, savepath * ".png", width=2560, height=1440)
            end
            return line_plot
        else
            print("Dimensions are not appropriate for plotting.")
            return
        end
    elseif tstate.model[1].in_dims == 2
        if tstate.model[end].out_dims == 1
            x = range(-1, stop=1, length=N)
            x = dtype.(collect(x))
            y = range(-1, stop=1, length=N)
            y = dtype.(collect(y))

            z = [tstate.model([x (yj .* dtype.(ones(length(x))))]', tps, tss)[1][1,:] for yj in y]
            z = reshape(vcat(z...),(N,N))

            if color_scheme == "teacher"
                if isnothing(title)
                    title="Teacher"
                end
                surface_plot = PlotlyJS.surface(
                    x = x,
                    y = y,
                    z = z,
                    colorscale = "Greys",
                    opacity = opacity,
                    name = title,
                    colorbar = attr(title=title, x=-0.15)
                )
            else
                if isnothing(title)
                    title="Student"
                end
                surface_plot = PlotlyJS.surface(
                    x = x,
                    y = y,
                    z = z,
                    colorscale = "Jet",
                    opacity = opacity,
                    name = title,
                    colorbar = attr(title=title)
                )
            end
            surface_plot = combine(combine_with_the_plot, surface_plot)
            layout = PlotlyJS.Layout(
                title = "Surface plot of model(x, y)",
                scene = Dict(
                    :xaxis => attr(title="X", showticklabels=showticklabels),
                    :yaxis => attr(title="Y", showticklabels=showticklabels),
                    :zaxis => attr(title="Z", showticklabels=showticklabels),
                    :aspectmode => "cube",
                    :camera => attr(
                        eye = attr(x=camera[1], y=camera[2], z=camera[3]),    # Camera position
                        center = attr(x=0, y=0, z=0),        # Look at point
                        up = attr(x=0, y=0, z=1)             # Up direction
                    ),
                ),
                showlegend = true,
                legend = attr(x = 0.5,
                 y = 1,
                 xanchor = "center",
                 yanchor = "top")
            )
            p = PlotlyJS.plot(surface_plot, layout)
            if display_plot
                display(p)
            end
            if !isnothing(savepath)
                PlotlyJS.savefig(p, savepath * ".png", width=2560, height=1440)
            end
            return surface_plot
        else
            print("Dimensions are not appropriate for plotting.")
            return
        end
    else
        print("Dimensions are not appropriate for plotting.")
        return
    end
    return
end

"""
    plot_data_teacher_and_student(tstate,teacher_tstate, train_set; display_plot=true)

    This function combines the above functions to produce a plot, in which the teacher function, the student function and the data sampled from the teacher are overlayed into one combined plot.
"""
function plot_data_teacher_and_student(tstate,teacher_tstate, train_set; display_plot=true, camera=(1.5,-1.5,1.5), savepath=nothing,markersizeattribute=2, showticklabels=false)
    if length(train_set)==1
        train_set_plot = net_scatter_plot(train_set[1]; show_legend=true, display_plot=display_plot,camera=camera,markersizeattribute=markersizeattribute, showticklabels=showticklabels)
    else
        train_set_plot = net_scatter_plot(train_set[1]; show_legend=false, display_plot=display_plot,camera=camera,markersizeattribute=markersizeattribute, showticklabels=showticklabels)
        for i in 2:length(train_set)-1
            train_set_plot = net_scatter_plot(train_set[i]; combine_with_the_plot=train_set_plot, show_legend=false, display_plot=display_plot,camera=camera,markersizeattribute=markersizeattribute, showticklabels=showticklabels)
        end
        train_set_plot = net_scatter_plot(train_set[end]; combine_with_the_plot=train_set_plot, display_plot=display_plot,camera=camera,markersizeattribute=markersizeattribute, showticklabels=showticklabels)
    end

    teacher_and_data_plot = net_smooth_plot(teacher_tstate; N=200, display_plot=display_plot, color_scheme="teacher", opacity=1.0, combine_with_the_plot=train_set_plot,camera=camera, showticklabels=showticklabels)

    endplot = net_smooth_plot(tstate; N=200, display_plot=display_plot, color_scheme="student", opacity=0.6, combine_with_the_plot=teacher_and_data_plot,camera=camera,savepath=savepath, showticklabels=showticklabels)

    return endplot
end