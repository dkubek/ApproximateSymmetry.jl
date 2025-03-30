using Distributed
using ProgressMeter
using DelimitedFiles
using CSV
using DataFrames

"""
    ComputeTask

Represents a single computation task in the processing pipeline.

# Fields
- `instance`: The problem instance to solve
- `run_index`: Which run number this represents (for multiple runs per instance)
"""
struct ComputeTask
    instance::AbstractInstance
    run_index::Int
end

"""
    SolutionResult

Represents a computed solution with its associated metadata.

# Fields
- `solution`: The computed solution object
- `instance_id`: Identifier of the source instance
- `run_index`: Which run number this represents
- `method_name`: Name of the method used
- `method_version`: Version of the method used
- `graph_type`: Type of graph (if applicable)
- `additional_metadata`: Any other metadata as key-value pairs
"""
struct SolutionResult
    solution::Solution
    instance_id::String
    run_index::Int
    method_name::String
    method_version::String
    graph_type::String
    additional_metadata::Dict{Symbol, Any}
end

"""
    OutputFiles

Represents a collection of files written for a single solution.

# Fields
- `permutation_file`: Path to the permutation output file
- `metrics_file`: Path to the metrics output file
"""
struct OutputFiles
    permutation_file::String
    metrics_file::String
end

"""
    TaskChannel

Type alias for a remote channel of compute tasks with optional sentinel values.
"""
const TaskChannel = RemoteChannel{Channel{Union{ComputeTask, Nothing}}}

"""
    SolutionChannel

Type alias for a remote channel of solution results with optional sentinel values.
"""
const SolutionChannel = RemoteChannel{Channel{Union{SolutionResult, Nothing}}}

"""
    OutputCollection

Type alias for a collection of output file groups.
"""
const OutputCollection = Vector{OutputFiles}

"""
    create_task_channel(buffer_size::Int=32) -> TaskChannel

Create a new remote channel for compute tasks with the specified buffer size.
"""
function create_task_channel(buffer_size::Int=32)::TaskChannel
    return RemoteChannel(() -> Channel{Union{ComputeTask, Nothing}}(buffer_size))
end

"""
    create_solution_channel(buffer_size::Int=32) -> SolutionChannel

Create a new remote channel for solution results with the specified buffer size.
"""
function create_solution_channel(buffer_size::Int=32)::SolutionChannel
    return RemoteChannel(() -> Channel{Union{SolutionResult, Nothing}}(buffer_size))
end

"""
    produce_tasks(
        dataset::AbstractDataset, 
        task_channel::TaskChannel, 
        num_runs::Int,
        all_instances::Vector{<:AbstractInstance}
    ) -> Nothing

Producer function that generates computation tasks and puts them in the task channel.
Sends a sentinel value (nothing) to each worker when done.
"""
function produce_tasks(
    dataset::AbstractDataset, 
    task_channel::TaskChannel, 
    num_runs::Int,
    all_instances::Vector{<:AbstractInstance}
) :: Nothing
    try
        # Generate all instance/run pairs as tasks
        for instance in all_instances
            for run in 1:num_runs
                task = ComputeTask(instance, run)
                put!(task_channel, task)
            end
        end
    finally
        # Signal end of tasks with sentinel values (one per worker)
        for _ in 1:length(workers())
            put!(task_channel, nothing)  # Sentinel value
        end
    end
    return nothing
end

"""
    consume_tasks(
        task_channel::TaskChannel,
        solution_channel::SolutionChannel,
        method::AbstractMethod,
        progress::Progress
    ) -> Nothing

Consumer function that takes tasks from the task channel, computes solutions,
and puts the results in the solution channel.
"""
function consume_tasks(
    task_channel::TaskChannel,
    solution_channel::SolutionChannel,
    method::AbstractMethod,
    progress::Progress
) :: Nothing
    try
        # Process tasks until we receive a sentinel value
        while true
            # Get next task
            task_or_sentinel = take!(task_channel)
            
            # Check for sentinel value
            if task_or_sentinel === nothing
                break
            end
            
            # Unpack task
            task = task_or_sentinel::ComputeTask
            
            try
                # Compute solution
                solution = Methods.solve(method, task.instance)
                
                # Extract graph type from instance metadata if available
                graph_type = "unknown"
                additional_metadata = Dict{Symbol, Any}()
                
                if hasproperty(task.instance, :metadata) && !isnothing(task.instance.metadata)
                    metadata = task.instance.metadata
                    if haskey(metadata, :graph_type)
                        graph_type = string(metadata[:graph_type])
                        # Remove graph_type to avoid duplication
                        metadata_copy = copy(metadata)
                        delete!(metadata_copy, :graph_type)
                        additional_metadata = metadata_copy
                    else
                        additional_metadata = metadata
                    end
                end
                
                # Create structured result object
                result = SolutionResult(
                    solution,
                    task.instance.id,
                    task.run_index,
                    method.name,
                    method.version,
                    graph_type,
                    additional_metadata
                )
                
                # Put solution in channel
                put!(solution_channel, result)
                
                # Update progress
                next!(progress)
            catch e
                @error "Error computing solution" instance=task.instance.id run=task.run_index exception=(e, catch_backtrace())
            end
        end
    finally
        # If this is the last worker to finish, send sentinel to solution channel
        # if isempty(task_channel)
        #     put!(solution_channel, nothing)  # Sentinel value
        # end
    end
    return nothing
end

"""
    consume_solutions(
        solution_channel::SolutionChannel,
        output_dir::String,
        progress::Progress
    ) -> OutputCollection

Consumer function that takes solution results from the solution channel,
writes them to disk, and returns a collection of output file paths.
"""
function consume_solutions(
    solution_channel::SolutionChannel,
    output_dir::String,
    progress::Progress
) :: OutputCollection
    output_files = OutputCollection()
    
    try
        # Process solutions until we receive a sentinel value
        while true
            # Get next solution result
            result_or_sentinel = take!(solution_channel)
            
            # Check for sentinel value
            if result_or_sentinel === nothing
                break
            end
            
            # Unpack result
            result = result_or_sentinel::SolutionResult
            
            try
                # Write solution to disk
                files = write_solution_files(result, output_dir)
                push!(output_files, files)
                
                # Update progress
                next!(progress)
            catch e
                @error "Error writing solution" instance_id=result.instance_id run=result.run_index exception=(e, catch_backtrace())
            end
        end
    catch e
        @error "Error in solution consumer" exception=(e, catch_backtrace())
    end
    
    return output_files
end

"""
    write_solution_files(
        result::SolutionResult,
        output_dir::String
    ) -> OutputFiles

Write a solution to disk and return the paths of written files.
"""
function write_solution_files(
    result::SolutionResult,
    output_dir::String
) :: OutputFiles
    # Create output directories
    solution_dir = joinpath(output_dir, result.method_name, result.method_version, result.graph_type)
    permutations_dir = joinpath(solution_dir, "Permutations")
    metrics_dir = joinpath(solution_dir, "Metrics")
    
    mkpath(permutations_dir)
    mkpath(metrics_dir)
    
    # File paths
    permutation_file = joinpath(permutations_dir, "$(result.instance_id)_run$(result.run_index).csv")
    metrics_file = joinpath(metrics_dir, "$(result.instance_id)_run$(result.run_index)_metrics.csv")
    
    # Write files
    write_permutation([result.solution], permutation_file)
    write_metrics([result.solution], metrics_file)
    
    return OutputFiles(permutation_file, metrics_file)
end

"""
    write_metrics(
        solutions::Vector{Solution{T}}, 
        filename::String
    ) -> String where T <: Number

Write solution metrics to a CSV file and return the file path.
"""
function write_metrics(
    solutions::Vector{Solution{T}}, 
    filename::String
) :: String where T <: Number
    all_metrics = []
    for (i, solution) in enumerate(solutions)
        solution_metrics = get_metrics(solution)
        # Add a run identifier to each metric
        # metrics_with_run = Dict{Symbol, Any}(:run => i)
        # for (metric, value) in solution_metrics
        #     metrics_with_run[metric] = value
        # end
        push!(all_metrics, solution_metrics)
    end

    metrics_df = DataFrame(all_metrics)
    CSV.write(filename, metrics_df)
    
    return filename
end

"""
    write_permutation(
        solutions::Vector{Solution{T}}, 
        filename::String
    ) -> String where T <: Number

Write solution permutations to a CSV file and return the file path.
"""
function write_permutation(
    solutions::Vector{Solution{T}}, 
    filename::String
) :: String where T <: Number
    n = size(solutions |> first |> get_result, 1)

    permutations = Vector{Vector{Int}}()
    for solution in solutions
        P = get_result(solution)
        permutation = [findfirst(x -> x == 1, P[i, :]) for i in 1:n] .- 1
        push!(permutations, permutation)
    end

    data = reduce(hcat, permutations)
    # headers = ["run$i" for i in eachindex(permutations)]

    open(filename, "w") do io
        # writedlm(io, [headers], ',')  
        writedlm(io, data, ',')
    end
    
    return filename
end

"""
    process_dataset(
        dataset::AbstractDataset, 
        method::AbstractMethod, 
        output_dir::String; 
        num_runs::Int=5, 
        force_recompute::Bool=false,
        task_buffer_size::Int=32,
        solution_buffer_size::Int=32
    ) -> OutputCollection

Process all instances in a dataset using the specified method and save results.
Uses a distributed producer/consumer architecture for efficient parallel processing.

# Arguments
- `dataset`: The dataset containing instances to process
- `method`: The method to use for solving instances
- `output_dir`: Directory where results will be saved
- `num_runs`: Number of solutions to compute for each instance
- `force_recompute`: Whether to recompute solutions that already exist
- `task_buffer_size`: Size of the task channel buffer
- `solution_buffer_size`: Size of the solution channel buffer

# Returns
- Collection of output file paths for all written solutions
"""
function process_dataset(
    dataset::AbstractDataset, 
    method::AbstractMethod, 
    output_dir::String; 
    num_runs::Int=5, 
    force_recompute::Bool=false,
    task_buffer_size::Int=32,
    solution_buffer_size::Int=32
) :: OutputCollection
    # Ensure output directory exists
    mkpath(output_dir)
    
    # Skip existing files if not forcing recomputation
    all_instances = collect(Datasets.instances(dataset))
    if !force_recompute
        # Filter instances with existing results
        # This is a placeholder - actual implementation would depend on your file naming scheme
        # all_instances = filter(instance -> !all_solutions_exist(instance, method, output_dir, num_runs), all_instances)
    end
    
    # If all instances are already processed, return early
    if isempty(all_instances)
        @info "All instances already processed. Use force_recompute=true to recompute."
        return OutputCollection()
    end
    
    # Create communication channels
    task_channel = create_task_channel(task_buffer_size)
    solution_channel = create_solution_channel(solution_buffer_size)
    
    # Calculate total tasks for progress reporting
    total_tasks = length(all_instances) * num_runs
    
    # Create progress meters
    task_progress = Progress(
        total_tasks, 
        desc="Computing solutions: ", 
        barglyphs=BarGlyphs("[=> ]"),
        barlen=50,
        offset=0,
    )
    
    solution_progress = Progress(
        total_tasks, 
        desc="Writing solutions: ", 
        barglyphs=BarGlyphs("[=> ]"),
        barlen=50,
        offset=1,
    )
    
    # Start the solution consumer (runs on main process)
    solution_consumer_task = @async consume_solutions(solution_channel, output_dir, solution_progress)
    
    # Start task consumers on worker processes
    worker_tasks = []
    for worker in workers()
        task = remotecall(consume_tasks, worker, task_channel, solution_channel, method, task_progress)
        push!(worker_tasks, task)
    end
    
    # Produce tasks (runs on main process)
    produce_tasks(dataset, task_channel, num_runs, all_instances)
    
    # Wait for all worker tasks to complete
    for t in worker_tasks
        fetch(t)
    end
    
    # Signal end of solutions AFTER all workers are done
    put!(solution_channel, nothing)
    
    # Wait for solution consumer to finish
    output_files = fetch(solution_consumer_task)

    ProgressMeter.finish!(task_progress)
    ProgressMeter.finish!(solution_progress)
    
    # Create summary files if needed
    if !isempty(output_files)
        create_summary_files(output_files, method, output_dir)
    end
    
    return output_files
end

"""
    create_summary_files(
        output_files::OutputCollection,
        method::AbstractMethod,
        output_dir::String
    ) -> Vector{String}

Create summary files aggregating metrics across all solutions.
"""
function create_summary_files(
    output_files::OutputCollection,
    method::AbstractMethod,
    output_dir::String
) :: String
    extract_properties = [("nNodes", Int), ("density", Int), ("sim", Int), ("run", Int)]

    instance_data = Dict{String, Vector{Any}}()
    metrics_data = DataFrame()
    for output_file in output_files
        basename_parts = split(basename(output_file.metrics_file), "_")
        for (property, type) in extract_properties
            property_index = findfirst(startswith(property), basename_parts)
            str_value = chopprefix(basename_parts[property_index], property)
            value = parse(type, str_value)

            values = get!(instance_data, property, type[])
            append!(values, value)
        end

        metrics = CSV.read(output_file.metrics_file, DataFrame)
        append!(metrics_data, metrics)
    end

    df = hcat(DataFrame(instance_data), metrics_data)
    println(df)

    summary_file = joinpath(output_dir, method.name, method.version, "summary.csv")
    CSV.write(summary_file, df)

    return summary_file
end


export process_dataset, write_solution, write_metrics, write_permutation,
       ComputeTask, SolutionResult, OutputFiles