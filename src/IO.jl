module IO

using ..Interfaces
using ..Datasets
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
include("io/readers/npz_loader.jl")
include("io/readers/csv_reader.jl")

end # module
