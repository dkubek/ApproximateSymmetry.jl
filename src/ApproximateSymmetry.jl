module ApproximateSymmetry

using Reexport

include("Common/Common.jl")
@reexport using .Common

include("Instances.jl")
@reexport using .Instances

include("Solutions/Solutions.jl")
@reexport using .Solutions

include("Methods/Methods.jl")
@reexport using .Methods

include("Datasets/Datasets.jl")
@reexport using .Datasets

# Convenience re-exports
export solve, adjacency, process_dataset

# CLI entry point
function main()
        CLI.main()
end

end # module
