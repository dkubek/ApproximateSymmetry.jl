module Interfaces

using LinearAlgebra

include("interfaces/instance/base.jl")
include("interfaces/instance/matrix_instance.jl")

include("interfaces/method.jl")

include("interfaces/output_format.jl")

include("interfaces/solution/base.jl")
include("interfaces/solution/solution.jl")

include("interfaces/dataset.jl")

end # module
