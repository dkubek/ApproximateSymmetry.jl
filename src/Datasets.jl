module Datasets

using ..Interfaces
using NPZ
using DataFrames

include("datasets/base.jl")

include("datasets/pidnebesna/instance/multiple_simulation_instance.jl")
include("datasets/pidnebesna/base.jl")

end # module
