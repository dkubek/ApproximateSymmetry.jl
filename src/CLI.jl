module CLI

using ..Interfaces
using ..Datasets
using ..IO
using ..Methods
using ArgParse

# Exports
export main, process_dataset

# Include implementation files
include("cli/app.jl")
include("cli/process.jl")

end # module
