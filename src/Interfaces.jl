module Interfaces

using LinearAlgebra

# Exports
export AbstractInstance, AbstractDataset, AbstractMethod, AbstractMetric,
        AbstractOutputFormat, MatrixInstance,
        AbstractSolution, Solution,
        get_result, get_metric, has_metric, set_metric!, get_metrics, supported_metrics

# Include implementation files
include("interfaces/abstractions.jl")
include("interfaces/instances.jl")
include("interfaces/solutions.jl")

end # module
