module Instances

using ..Common

"""
	AbstractInstance

Abstract supertype for all instance types in the system.
An instance represents a single problem to be solved.
"""
abstract type AbstractInstance end

function adjacency(instance::AbstractInstance)
        error("adjacency not implemented for $(typeof(instance))")
end

function load!(instance::AbstractInstance)
        error("load! not implemented for $(typeof(instance))")
end

export AbstractInstance, adjacency

end # module
