module Common

using LinearAlgebra

# Core abstractions
abstract type ObjectiveFunction end

"""
	BufferManager{T}

Manages reusable matrix and vector buffers for efficient computation.
"""
mutable struct BufferManager{T <: Real}
	matrix_buffers::Dict{Symbol, Matrix{T}}
	vector_buffers::Dict{Symbol, Vector{T}}
end

BufferManager{T}() where {T <: Real} = BufferManager{T}(Dict{Symbol, Matrix{T}}(), Dict{Symbol, Vector{T}}())

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


"""
    to_permutation_matrix(perm::AbstractVector{Int})

Create an efficient representation of a permutation matrix based on the input size.
Uses sparse matrices for large permutations.
"""
function to_permutation_matrix(perm::AbstractVector{Int})
    n = length(perm)
    
    # Use sparse matrix for large permutations
    if n > 100
        I = 1:n
        return sparse(I, perm, ones(Bool, n), n, n)
    else
        P = zeros(Bool, n, n)
        @inbounds for (i, j) in enumerate(perm)
            P[i, j] = true
        end
        return P
    end
end


# Export public interface
export ObjectiveFunction, BufferManager
export is_symmetric_matrix, ApproximateSymmetryError
export to_permutation_matrix

end # module