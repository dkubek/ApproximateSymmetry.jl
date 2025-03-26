struct PidnebesnaInstance{T} <: AbstractInstance
    id::String
    simulations::Dict{Int,MatrixInstance}
    metadata::Dict{Symbol,Any}
end

"""
Construct a matrix instance with the given ID and matrix data.
"""
PidnebesnaInstance(id::String, simulations::Dict{Int,MatrixInstance}) where {T} =
    PidnebesnaInstance{T}(id, simulations, Dict{Symbol,Any}())


export PidnebesnaInstance

