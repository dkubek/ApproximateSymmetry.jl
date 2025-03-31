using LinearAlgebra

"""
	MatrixInstance{T<:Real}

Represents a matrix-based problem instance with type-stable matrix elements.

# Fields
- `id::String`: Unique identifier for the instance
- `matrix::Matrix{T}`: The matrix data (typically an adjacency matrix)
- `n::Int`: Size of the matrix (number of nodes)
- `path::String`: Path to the source file if loaded from disk
- `metadata::Dict{Symbol,Any}`: Additional metadata about the instance
"""
struct MatrixInstance{T <: Real} <: AbstractInstance
	id::String
	matrix::Matrix{T}
	n::Int
	metadata::Dict{Symbol, Any}

	function MatrixInstance(id::String, matrix::Matrix{T}) where {T <: Real}
		n, m = size(matrix)
		if n != m
			throw(Common.ApproximateSymmetryError("Matrix must be square: got dimensions ($n,$m)"))
		end

		new{T}(id, matrix, n, Dict{Symbol, Any}())
	end
end

function adjacency(instance::MatrixInstance{T}) where {T <: Real}
	return instance.matrix
end

export MatrixInstance, adjacency
