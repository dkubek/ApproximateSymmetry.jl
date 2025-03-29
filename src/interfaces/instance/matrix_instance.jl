"""
Represents a matrix-based problem instance with type-stable matrix elements.
"""
struct MatrixInstance{T} <: AbstractInstance
    id::String
    matrix::Matrix{T}

    n::Int
    path::String
end

"""
Construct a matrix instance with the given ID and matrix data.
"""
function MatrixInstance(id::String, matrix::Matrix{T}) where {T}
    (n, m) = size(matrix)
    m == n || throw("Matrix must be square!")

    MatrixInstance{T}(id, matrix, n, "")
end

"""
Return the size of the matrix in the instance.
"""
Base.size(instance::MatrixInstance) = size(instance.matrix)

"""
Return the size of the matrix in the instance along the specified dimension.
"""
Base.size(instance::MatrixInstance, dim::Int) = size(instance.matrix, dim)

"""
Access the underlying matrix with the given indices.
"""
Base.getindex(instance::MatrixInstance, indices...) = getindex(instance.matrix, indices...)


adjacency(instance::MatrixInstance) = instance.matrix

export MatrixInstance
