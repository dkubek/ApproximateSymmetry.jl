module IO

using ..Interfaces
using NPZ
using CSV
using DataFrames

# Exports
export read_dataset, read_solution, load_instance,
        write_solution, write_summary, CSVOutputFormat

# Include implementation files
include("io/readers/base.jl")
include("io/writers/base.jl")
include("io/writers/csv.jl")

end # module
