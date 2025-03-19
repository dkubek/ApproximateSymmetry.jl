module Datasets

using ..Interfaces
using NPZ
using DataFrames
using ProgressMeter

# Exports
export NPZDataset, load_dataset, process_dataset, iterate_instances

# Include implementation files
include("datasets/pidnebesna.jl")
include("datasets/base.jl")

end # module
