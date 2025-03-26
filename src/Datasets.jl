module Datasets

using ..Interfaces
using NPZ
using DataFrames
using ProgressMeter

# Exports
export NPZDataset, load_dataset, process_dataset, iterate_instances

# Include implementation files
include("datasets/base.jl")

include("datasets/pidnebesna/base.jl")
include("datasets/pidnebesna/loader.jl")
include("datasets/pidnebesna/process.jl")

end # module
