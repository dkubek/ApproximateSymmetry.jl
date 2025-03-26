"""
Read a dataset of the specified type from the given path.
Returns a dataset object of the requested type.
"""
function read_dataset(::Type{T}, path::String; kwargs...) where {T<:AbstractDataset}
    error("read_dataset not implemented for type $T")
end

"""
Read a solution from the given path using the specified format.
"""
function read_solution(path::String, format::AbstractOutputFormat, ::Type{Solution}; kwargs...)
    error("read_solution not implemented for format type $(typeof(format))")
end

# Export reader functions
export read_dataset, read_solution
