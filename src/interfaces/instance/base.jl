"""
    AbstractInstance

Abstract supertype for all instance types in the system.
An instance represents a single problem to be solved.
"""
abstract type AbstractInstance end

function adjacency(instance::AbstractInstance)
    error("adjacency not implemented for format type $(typeof(instance))")
end

export AbstractInstance, adjacency
