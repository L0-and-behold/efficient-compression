# this is the root module
module CompressingClassifiersMLPs

    include("TrainArgs.jl")
    include("Checkpointer.jl")
    
    include("OptimizationProcedures/OptimizationProcedures.jl")
    include("DatasetsModels/DatasetsModels.jl")
    include("TrainingTools.jl")
    include("Database/Database.jl")
    include("BatchRun/BatchRun.jl")

end # module compressing_classifiers_and_MLPs
