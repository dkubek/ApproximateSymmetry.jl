module CLI

using ..Core
using ..Datasets
using ..IO
using ..Methods
using ArgParse

# Exports
export main

# Include implementation files
include("cli/app.jl")

end # module
