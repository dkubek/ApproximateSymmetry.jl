"""
    AbstractMethod

Abstract supertype for all solution methods.
Methods implement algorithms to solve instances.
"""
abstract type AbstractMethod end

"""
    supported_metrics(method::AbstractMethod)

Return a vector of symbols representing metrics this method can compute.
"""
function supported_metrics(method::AbstractMethod)
    return Symbol[]
end


export AbstractMethod, supported_metrics
