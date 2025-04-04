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

# Include dataset implementations
include("pidnebesna.jl")

# Export public interface
export AbstractDataset

end # module
