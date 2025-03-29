module Common

using LinearAlgebra

# Core abstractions
abstract type ObjectiveFunction end

"""
    BufferManager{T}

Manages reusable matrix and vector buffers for efficient computation.
"""
mutable struct BufferManager{T<:Real}
    matrix_buffers::Dict{Symbol,Matrix{T}}
    vector_buffers::Dict{Symbol,Vector{T}}
end

BufferManager{T}() where {T<:Real} = BufferManager{T}(Dict{Symbol,Matrix{T}}(), Dict{Symbol,Vector{T}}())

# Trait functions
"""
    is_symmetric_matrix(x) -> Bool

Check if an object is a symmetric matrix.
"""
is_symmetric_matrix(x) = false
is_symmetric_matrix(x::AbstractMatrix) = issymmetric(x)

# Error types
struct ApproximateSymmetryError <: Exception
    msg::String
end

Base.showerror(io::IO, e::ApproximateSymmetryError) = print(io, "ApproximateSymmetryError: ", e.msg)

# Export public interface
export ObjectiveFunction, BufferManager
export is_symmetric_matrix, ApproximateSymmetryError

end # module