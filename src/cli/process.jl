using DelimitedFiles
using CSV
using DataFrames

# """
# Process all instances in an NPZ dataset using the specified method and save results.
# """
# function process_dataset(
#     dataset::NPZDataset, method::AbstractMethod, output_dir::String;
#     num_runs=5, force_recompute=false
# )
#     # Create output directories
#     method_dir = joinpath(output_dir, method.name)
#     version_dir = joinpath(method_dir, method.version)
# 
#     # Process each instance
#     # progress = Progress(length(collect(iterate_instances(dataset))), 1, "Processing instances...")
# 
#     for (graph_type, instances) in dataset.instances
#         model_output_dir = joinpath(version_dir, graph_type)
#         mkpath(model_output_dir)
# 
#         permutations_dir = joinpath(model_output_dir, "Permutations")
#         mkpath(permutations_dir)
# 
#         metrics_dir = joinpath(model_output_dir, "Metrics")
#         mkpath(metrics_dir)
# 
#         for instance in instances
#             for simulation in instance.simulations
#                 permutations_filename = joinpath(permutations_dir, "$(simulation.id).csv")
#                 metrics_filename = joinpath(metrics_dir, "$(simulation.id)_metrics.csv")
# 
#                 if !force_recompute && isfile(permutations_filename) && isfile(metrics_filename)
#                     continue
#                 end
# 
#                 solutions = Vector{Solution{Float64}}()
#                 for _ in 1:num_runs
#                     solution = solve(method, simulation)
#                     push!(solutions, solution)
#                 end
# 
#                 write_permutation(solutions, permutations_filename)
#                 write_metrics(solutions, metrics_filename)
# 
#                 # next!(progress)
#             end
#         end
#     end
# 
#     # Compile and save summaries for each metric
#     # for metric_name in supported_metrics(method)
#     #     summary_file = joinpath(model_output_dir, "$(method.name)_$(metric_name)_$(dataset.graph_type).csv")
#     #     write_summary(format, metrics_dir, summary_file; metric=metric_name, num_runs=num_runs)
#     # end
# 
#     return nothing
# end

"""
Write a solution to the specified path using CSV format.
"""
function write_solution(
    solutions::Vector{Solution{T}},
    instance::MatrixInstance,
    outdir::String
) where T <: Number

    permutations_dir = joinpath(outdir, "Permutations")
    mkpath(permutations_dir)
    permutations_filename = joinpath(permutations_dir, "$(instance.id).csv")
    write_permutation(solutions, permutations_filename)

    metrics_dir = joinpath(outdir, "Metrics")
    mkpath(metrics_dir)
    metrics_filename = joinpath(metrics_dir, "$(instance.id)_metrics.csv")
    write_metrics(solutions, metrics_filename)
end

write_solution(solution::Solution, instance::MatrixInstance, outdir::String) =
    write_solution([solution], instance, outdir)

function write_metrics(
    solutions::Vector{Solution{T}}, filename::String
) where T <: Number
    all_metrics = []
    for (i, solution) in enumerate(solutions)
        solution_metrics = get_metrics(solution)
        # Add a run identifier to each metric
        metrics_with_run = Dict{Symbol, Any}(:run => i)
        for (metric, value) in solution_metrics
            metrics_with_run[metric] = value
        end
        push!(all_metrics, metrics_with_run)
    end

    metrics_df = DataFrame(all_metrics)

    CSV.write(filename, metrics_df)
end

function write_permutation(
    solutions::Vector{Solution{T}}, filename::String
) where T <: Number
    n = size(solutions |> first |> get_result, 1)

    permutations = Vector{Vector{Int}}()
    for solution in solutions
        P = get_result(solution)
        permutation = [findfirst(x -> x == 1, P[i, :]) for i in 1:n] .- 1
        push!(permutations, permutation)
    end

    data = reduce(hcat, permutations)
    headers = ["run$i" for i in eachindex(permutations)]

    open(filename, "w") do io
        writedlm(io, [headers], ',')  
        writedlm(io, data, ',')
    end
end


function process_dataset(
    dataset::AbstractDataset, 
    method::Methods.AbstractMethod, 
    output_dir::String; 
    num_runs::Int=5, 
    force_recompute::Bool=false
)
    mkpath(output_dir)
    
    all_instances = collect(Datasets.instances(dataset))
    
    # Pre-allocate thread-local storage for results
    thread_results = [ Dict{String, Tuple{Vector{Solutions.Solution},Dict{Symbol,Any}}}() for _ in 1:Threads.nthreads() ]
    
    Threads.@threads for instance in all_instances
        thread_id = Threads.threadid()
        local_results = thread_results[thread_id]
        
        instance_solutions = Vector{Solutions.Solution}(undef, num_runs)
        for run in 1:num_runs
            solution = Methods.solve(method, instance)
            instance_solutions[run] = solution
        end
        
        # Store solutions for this instance
        local_results[instance.id] = (instance_solutions, copy(instance.metadata))
    end
    
    # Combine results from all threads
    combined_results = Dict{String, Tuple{Vector{Solutions.Solution},Dict{Symbol,Any}}}()
    for local_result in thread_results
        merge!(combined_results, local_result)
    end
    
    return combined_results
end

export process_dataset, write_solution
