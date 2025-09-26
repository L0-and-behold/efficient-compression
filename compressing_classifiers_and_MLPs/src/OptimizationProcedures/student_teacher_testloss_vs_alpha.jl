"""
A simple script to run, test and develop the different optimization procedures in the Teacher-Student Setup.
"""

using Revise
using CUDA
using JLD2

include("../TrainArgs.jl")
include("HelperFunctions/generate_networks_and_data.jl")
include("HelperFunctions/plot_networks_and_their_output.jl")

# Load the different L0_regularization procedures
include("RL1_procedure.jl")
include("DRR_procedure.jl")
include("PMMP_procedure.jl")
include("layerwise_procedure.jl")
# Note that if RL1 is initialized with alpha=0=rho, then it corresponds to unregularized ("vanilla") optimization

include("NormalFormCode/convert_flux_to_lux.jl")
include("NormalFormCode/symbolic_representation.jl")
include("NormalFormCode/random_networks.jl")

alphas = Float32.([0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1,1.5,2,5,10])
repetitions = 10

all_test_errs = []

# i=1
# alpha=alphas[end]
for alpha in alphas
  test_errs = Float32[]
  for i in 1:repetitions
    # Training arguments are initialized
    println()
    println("alpha=",alpha," i=",i)
    println()

    args = TrainArgs(; T=Float32) 

    args.gauss_loss = false
    args.α = alpha
    args.min_epochs = 1500
    args.finetuning_max_epochs = 10000
    args.val_set_size = 5000
    args.test_set_size = 10000

    # Teacher and Student models are initialized and data is generated from the Gaussian distribution, whose mean value is computed by the teacher network.
    train_set, validation_set, test_set, tstate, loss_fctn, args, teacher_tstate = setup_data_teacher_and_student(args; architecture_teacher=args.architecture_teacher, architecture_student=args.architecture_student, seed_teacher=35, seed_student=42+i, seed_train_set=i, seed_val_set=repetitions+i, seed_test_set=2*repetitions+i, teacher_weight_scaling=2, loss_fctn = Lux.MSELoss(), opt = Optimisers.Adam)

    scale_alpha_rho!(args, train_set, loss_fctn)
    
    tstate, logs, loss_fun = DRR_procedure(train_set, validation_set, test_set, tstate, loss_fctn, args);
    
    # Visalize Results
    pt = plot_weights(tstate; legend=nothing)
    Plots.savefig(pt, "test_error_vs_alpha_results/alpha_" * string(alpha) * " student_weights_" * string(i) * ".pdf")

    flux_network = convert_fully_connected_network_to_flux(tstate.parameters.p,tstate.model)
    normalized_flux_network = normal_form(flux_network; device=Flux.cpu)
    normalized_tstate = convert_to_tstate(normalized_flux_network)
    ptn = plot_weights(normalized_tstate)
    Plots.savefig(ptn, "test_error_vs_alpha_results/alpha_" * string(alpha) * " student_weights_normalform_" * string(i) * ".pdf")

    if alpha==0f0 && i==1
      flux_network_teacher = convert_fully_connected_network_to_flux(teacher_tstate.parameters,teacher_tstate.model)
      normalized_flux_network_teacher = normal_form(flux_network_teacher; device=Flux.cpu)
      normalized_tstate_teacher = convert_to_tstate(normalized_flux_network_teacher)
      ptt = plot_weights(normalized_tstate_teacher; legend=nothing)
      Plots.savefig(ptt, "test_error_vs_alpha_results/teacher_weights_normalform.pdf")
    end

    if isempty(logs["turning_points_val_loss"])
      push!(test_errs, logs["test_loss"][end][2])
    else
      comparison_test_losses = [tuple[2] for tuple in logs["test_loss"][args.logs["turning_points_val_loss"]]]
      push!(test_errs, minimum(comparison_test_losses))
    end

    pdts = plot_data_teacher_and_student(tstate, teacher_tstate, train_set, savepath="test_error_vs_alpha_results/alpha_" * string(alpha) * " student_teacher_functions_" * string(i), markersizeattribute=5)

  end
  push!(all_test_errs, test_errs)
end

@save "all_test_errs.jld2" all_test_errs

