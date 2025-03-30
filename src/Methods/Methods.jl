module Methods

using LinearAlgebra
using ..Common
using ..Instances
using ..Solutions

"""
    AbstractMethod

Abstract supertype for all solution methods.
Methods implement algorithms to solve instances.
"""
abstract type AbstractMethod end

"""
    MethodParameters

Container for method parameters with typed access.

# Fields
- `values::Dict{Symbol,Any}`: Dictionary of parameter values
"""
struct MethodParameters
    values::Dict{Symbol,Any}
end

MethodParameters() = MethodParameters(Dict{Symbol,Any}())

function solve(method::AbstractMethod, instance::Instances.AbstractInstance)
    error("solve not implemented for $(typeof(method)) and $(typeof(instance))")
end

function supported_metrics(method::AbstractMethod)
    return Symbol[:time]
end

include("objective_function.jl")

include("simple_method.jl")
include("irwin_hall.jl")

export AbstractMethod, MethodParameters, solve, supported_metrics

end # module