using Distributed
using Random

@everywhere begin
    import Pkg
    Pkg.activate(@__DIR__)
    # Pkg.instantiate()
    # Pkg.precompile()
end

@everywhere include("src/process.jl")

# MAIN SCRIPT
# -----------

using ArgParse
using ProgressMeter

function main()
    data_dir = joinpath(@__DIR__, "data/pidnebesna/")
    result_dir = joinpath(@__DIR__, "results")

    total = PidnebesnaDataset(data_dir) |> length
    ds = PidnebesnaDataset(data_dir) |> load!
    #method = Methods.SimpleMethod("SimpleMethod", "v1")
    method = Methods.IHMethod()
    #p = ProgressMeter.Progress(total)
    p = ProgressMeter.Progress(10)
    all_instances = ds |> collect
    Threads.@threads for i in 1:10
        process(all_instances[i], method, result_dir; nruns=5)
        ProgressMeter.next!(p)
    end


    #@showprogress pmap(collect(1:total)) do i
    #    # Get the instance by index (avoids collecting everything)
    #    ds = PidnebesnaDataset(data_dir) |> load!
    #    #method = Methods.SimpleMethod("SimpleMethod", "v1")
    #    method = Methods.IHMethod()

    #    instance = iterate(Iterators.drop(ds, i - 1))[1]
    #    result = process(instance, method, result_dir; nruns=5)
    #    return nothing
    #end

end

"""
    get_task_count(dataset_type::String, num_runs::Int) -> Int

Estimate the total number of tasks for a given dataset and number of runs.
"""
function get_task_count(dataset::AbstractDataset, num_runs::Int)
    instances = collect(Datasets.instances(dataset))
    return length(instances) * num_runs
end

# function main()
#     args = parse_commandline()
# 
#     # Print banner
#     println("""
#     ╔═══════════════════════════════════════════════╗
#     ║ ApproximateSymmetry Parallel Task Runner      ║
#     ╚═══════════════════════════════════════════════╝
#     """)
# 
#     # Check if ApproximateSymmetry is available
#     try
#         @eval using ApproximateSymmetry
#     catch e
#         if isa(e, ArgumentError)
#             println("Error: ApproximateSymmetry package not found.")
#             println("Make sure you're running this script from the ApproximateSymmetry project directory.")
#             println("Try: cd /path/to/ApproximateSymmetry && julia scripts/parallel_runner.jl")
#             exit(1)
#         else
#             rethrow(e)
#         end
#     end
# 
#     # Print configuration
#     println("Configuration:")
#     println("  Dataset:   $(args["dataset"])")
#     println("  Method:    $(args["method"]) ($(args["version"]))")
#     println("  Output:    $(args["output"])")
#     println("  Runs:      $(args["runs"])")
#     println("  Processes: $(args["processes"])")
#     println("  Force:     $(args["force"])")
#     println("  Dry run:   $(args["dry-run"])")
#     println()
# 
#     success = run_parallel_tasks(args)
#     exit(success ? 0 : 1)
# end



# Run the main function
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
