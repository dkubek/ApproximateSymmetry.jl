module ApproximateSymmetry

using LinearAlgebra
using NPZ
using CSV
using DataFrames
using ProgressMeter
using Dates
using Reexport
using ArgParse

# Export key types and functions
export
    # Abstract types
    AbstractInstance,
    AbstractDataset,
    AbstractMethod,
    AbstractOutputFormat,

    # Concrete instance types
    MatrixInstance,

    # Dataset types
    NPZDataset,

    # Solution types
    AbstractSolution,
    Solution,

    # Loading functions
    load_instance,
    read_dataset,
    read_solution,

    # Method interfaces
    solve,
    supported_metrics,

    # Output formatting
    write_solution,
    write_summary,
    CSVOutputFormat,

    # Dataset processing
    iterate_instances,
    process_dataset

# Include core abstractions
include("abstractions.jl")
include("instances.jl")
include("solutions.jl")

# Include dataset components first since others depend on them
include("datasets/pidnebesna.jl")  # NPZDataset is defined here
include("datasets/base.jl")

# Include IO components
include("io/readers/base.jl")
include("io/writers/base.jl")

include("io/writers/csv.jl")
include("io/readers/npz_loader.jl")
include("io/readers/csv_reader.jl")  # Add the CSV reader

# Include method components
include("methods/base.jl")

# Include formulations
include("formulations/abstract_formulation.jl")
include("formulations/common.jl")
include("formulations/objective_functions.jl")

# Include CLI
include("cli/app.jl")

end
