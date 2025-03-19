module Methods

using ..Interfaces
using LinearAlgebra

# Exports
export solve, SimpleMethod, set_parameter!, get_parameter, add_supported_metric!

# Include implementation files
include("methods/base.jl")

end # module
