module Datasets

using ..Common
using ..Instances
using ..Methods
using ..Solutions

"""
    AbstractDataset

Abstract supertype for collections of instances.
A dataset provides a consistent interface for accessing multiple instances.
"""
abstract type AbstractDataset end

function instances(dataset::AbstractDataset)
    error("instances not implemented for $(typeof(dataset))")
end

function count_instances(dataset::AbstractDataset)
    return length(collect(instances(dataset)))
end


# Include dataset implementations
include("pidnebesna.jl")

# Export public interface
export AbstractDataset, instances, count_instances, filter_instances, process_dataset

end # module