"""
    MatrixInstance <: AbstractInstance

Represents a matrix-based problem instance.

# Fields
- `id::String`: Unique identifier for the instance
- `matrix::Matrix`: The matrix data
- `metadata::Dict`: Additional instance metadata
"""
struct MatrixInstance <: AbstractInstance
    id::String
    matrix::Matrix
    metadata::Dict{Symbol,Any}
end

"""
    MatrixInstance(id::String, matrix::Matrix)

Construct a matrix instance with the given ID and matrix data.
"""
MatrixInstance(id::String, matrix::Matrix) = MatrixInstance(id, matrix, Dict{Symbol,Any}())

"""
    Base.size(instance::MatrixInstance)

Return the size of the matrix in the instance.
"""
Base.size(instance::MatrixInstance) = size(instance.matrix)

"""
    Base.size(instance::MatrixInstance, dim::Int)

Return the size of the matrix in the instance along the specified dimension.
"""
Base.size(instance::MatrixInstance, dim::Int) = size(instance.matrix, dim)

"""
    Base.getindex(instance::MatrixInstance, indices...)

Access the underlying matrix with the given indices.
"""
Base.getindex(instance::MatrixInstance, indices...) = getindex(instance.matrix, indices...)

"""
    get_metadata(instance::MatrixInstance, key::Symbol, default=nothing)

Retrieve metadata from the instance with an optional default value.
"""
function get_metadata(instance::MatrixInstance, key::Symbol, default=nothing)
    get(instance.metadata, key, default)
end

"""
    set_metadata!(instance::MatrixInstance, key::Symbol, value)

Set metadata for the instance.
"""
function set_metadata!(instance::MatrixInstance, key::Symbol, value)
    instance.metadata[key] = value
    return instance
end
