using ProgressMeter
using DelimitedFiles
using CSV
using DataFrames

# Type definitions --------------------------------------------------------------

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
    additional_metadata::Dict{Symbol,Any}
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

const OutputCollection = Vector{OutputFiles}

# Filename utilities ------------------------------------------------------------

"""
    get_output_filenames(result::SolutionResult, output_dir::String, dir_paths::Dict{String,Dict{String,String}})
    -> Tuple{String,String}

Generate the filenames for permutation and metrics files based on the result.
"""
function get_output_filenames(
    result::SolutionResult,
    output_dir::String,
    dir_paths::Dict{String,Dict{String,String}}
)::Tuple{String,String}
    # Get the paths for this graph type, or use unknown as fallback
    graph_paths = get(dir_paths, result.graph_type, dir_paths["unknown"])

    # File paths
    permutation_file = joinpath(graph_paths["permutations"],
        "$(result.instance_id)_run$(result.run_index).csv")
    metrics_file = joinpath(graph_paths["metrics"],
        "$(result.instance_id)_run$(result.run_index)_metrics.csv")

    return permutation_file, metrics_file
end

"""
    result_exists(result::SolutionResult, output_dir::String, dir_paths::Dict{String,Dict{String,String}})
    -> Bool

Check if the output files for a result already exist.
"""
function result_exists(
    result::SolutionResult,
    output_dir::String,
    dir_paths::Dict{String,Dict{String,String}}
)::Bool
    permutation_file, metrics_file = get_output_filenames(result, output_dir, dir_paths)
    return isfile(permutation_file) && isfile(metrics_file)
end

"""
    should_process_task(task::ComputeTask, method::AbstractMethod, output_dir::String, dir_paths::Dict{String,Dict{String,String}}, force_recompute::Bool)
    -> Bool

Determine if a task should be processed based on whether its results already exist and the force_recompute flag.
"""
function should_process_task(
    task::ComputeTask,
    method::AbstractMethod,
    output_dir::String,
    dir_paths::Dict{String,Dict{String,String}},
    force_recompute::Bool
)::Bool
    # If force_recompute is true, always process
    if force_recompute
        return true
    end

    # Create a dummy result to check if files exist
    # We need to extract the graph type from metadata
    graph_type = "unknown"
    if hasproperty(task.instance, :metadata) &&
       !isnothing(task.instance.metadata) &&
       haskey(task.instance.metadata, :graph_type)
        graph_type = string(task.instance.metadata[:graph_type])
    end

    dummy_result = SolutionResult(
        # We don't need a real solution, just the metadata fields
        # used for filename generation
        nothing,
        task.instance.id,
        task.run_index,
        method.name,
        method.version,
        graph_type,
        Dict{Symbol,Any}()
    )

    # Check if result exists
    return !result_exists(dummy_result, output_dir, dir_paths)
end

# Core computation function ----------------------------------------------------

"""
    extract_metadata(instance::AbstractInstance) -> Tuple{String,Dict{Symbol,Any}}

Extract metadata from an instance with optimized copying.
"""
function extract_metadata(instance::AbstractInstance)::Tuple{String,Dict{Symbol,Any}}
    graph_type = "unknown"
    additional_metadata = Dict{Symbol,Any}()

    if hasproperty(instance, :metadata) && !isnothing(instance.metadata)
        metadata = instance.metadata
        if haskey(metadata, :graph_type)
            graph_type = string(metadata[:graph_type])
            # More efficient selective copy
            for (k, v) in metadata
                if k != :graph_type
                    additional_metadata[k] = v
                end
            end
        else
            # Direct reference if no modifications needed
            additional_metadata = metadata
        end
    end

    return graph_type, additional_metadata
end

"""
    solve_task(task::ComputeTask, method::AbstractMethod) -> SolutionResult

Solve a single computation task and return a solution result.
"""
function solve_task(task::ComputeTask, method::AbstractMethod)::SolutionResult
    # Compute solution
    solution = Methods.solve(method, task.instance)

    # Extract metadata using optimized function
    graph_type, additional_metadata = extract_metadata(task.instance)

    # Create result
    return SolutionResult(
        solution,
        task.instance.id,
        task.run_index,
        method.name,
        method.version,
        graph_type,
        additional_metadata
    )
end

# Directory management ---------------------------------------------------------

"""
    ensure_directories(output_dir::String, method::AbstractMethod, graph_types::Set{String}=Set(["unknown"]))
    -> Dict{String,Dict{String,String}}

Create all required directories for output files.
Returns a mapping of graph_type -> dict of directory paths.
"""
function ensure_directories(
    output_dir::String,
    method::AbstractMethod,
    graph_types::Set{String}=Set(["unknown"])
)::Dict{String,Dict{String,String}}
    # Create main directories
    base_dir = joinpath(output_dir, method.name, method.version)
    try
        mkpath(base_dir)
    catch e
        error("Failed to create base directory: $base_dir. Error: $e")
    end

    # Create a nested dictionary of paths for quick lookup
    dir_paths = Dict{String,Dict{String,String}}()

    # Ensure "unknown" is always included
    if !("unknown" in graph_types)
        push!(graph_types, "unknown")
    end

    for graph_type in graph_types
        dir_paths[graph_type] = Dict{String,String}()

        # Create type-specific directories
        solution_dir = joinpath(base_dir, graph_type)
        permutations_dir = joinpath(solution_dir, "Permutations")
        metrics_dir = joinpath(solution_dir, "Metrics")

        try
            mkpath(permutations_dir)
            mkpath(metrics_dir)
        catch e
            error("Failed to create directories for graph type $graph_type. Error: $e")
        end

        dir_paths[graph_type]["base"] = solution_dir
        dir_paths[graph_type]["permutations"] = permutations_dir
        dir_paths[graph_type]["metrics"] = metrics_dir
    end

    return dir_paths
end

# File IO functions ------------------------------------------------------------

"""
    write_solution_files(
        result::SolutionResult,
        output_dir::String,
        dir_paths::Dict{String,Dict{String,String}}
    ) -> OutputFiles

Write a solution to disk and return the paths of written files.
"""
function write_solution_files(
    result::SolutionResult,
    output_dir::String,
    dir_paths::Dict{String,Dict{String,String}}
)::OutputFiles
    # Get filenames
    permutation_file, metrics_file = get_output_filenames(result, output_dir, dir_paths)

    # Write files
    try
        write_permutation([result.solution], permutation_file)
        write_metrics([result.solution], metrics_file)
    catch e
        @error "Failed to write solution files" result.instance_id result.run_index exception = e
        # Attempt to clean up partial files
        isfile(permutation_file) && rm(permutation_file, force=true)
        isfile(metrics_file) && rm(metrics_file, force=true)
        rethrow(e)
    end

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
)::String where {T<:Number}
    # Pre-allocate for expected size
    all_metrics = Vector{Dict{Symbol,Any}}(undef, length(solutions))

    for (i, solution) in enumerate(solutions)
        all_metrics[i] = get_metrics(solution)
    end

    metrics_df = DataFrame(all_metrics)

    try
        CSV.write(filename, metrics_df)
    catch e
        @error "Failed to write metrics file" filename exception = e
        rethrow(e)
    end

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
)::String where {T<:Number}
    # Get matrix dimensions
    n = size(solutions |> first |> get_result, 1)
    num_solutions = length(solutions)

    # Pre-allocate for better performance
    permutation_matrix = Matrix{Int}(undef, n, num_solutions)

    # Extract permutations efficiently
    for (sol_idx, solution) in enumerate(solutions)
        P = get_result(solution)
        # Fill directly into matrix
        for i in 1:n
            permutation_matrix[i, sol_idx] = findfirst(x -> x == 1, view(P, i, :)) - 1
        end
    end

    try
        open(filename, "w") do io
            writedlm(io, permutation_matrix, ',')
        end
    catch e
        @error "Failed to write permutation file" filename exception = e
        rethrow(e)
    end

    return filename
end

# Summary generation ----------------------------------------------------------

"""
    create_summary_files(
        output_files::OutputCollection,
        method::AbstractMethod,
        output_dir::String
    ) -> String

Create summary files aggregating metrics across all solutions.
"""
function create_summary_files(
    output_files::OutputCollection,
    method::AbstractMethod,
    output_dir::String
)::String
    if isempty(output_files)
        @warn "No output files provided for summary generation"
        return ""
    end

    # Create structured approach to collect all data
    instance_data = Dict{String,Vector{Any}}()
    all_metrics = DataFrame()

    # Process output files in batches to limit memory usage
    batch_size = min(100, length(output_files))
    num_batches = ceil(Int, length(output_files) / batch_size)

    prog = Progress(length(output_files), desc="Generating summary:", dt=1.0)

    for batch_idx in 1:num_batches
        start_idx = (batch_idx - 1) * batch_size + 1
        end_idx = min(batch_idx * batch_size, length(output_files))

        batch_metrics = DataFrame()

        for i in start_idx:end_idx
            output_file = output_files[i]

            try
                # Extract graph type from directory structure
                graph_name = basename(dirname(dirname(output_file.metrics_file)))
                graph_names = get!(instance_data, "graph", String[])
                push!(graph_names, graph_name)

                # Parse filename properties more efficiently
                properties = extract_filename_properties(output_file.metrics_file)

                # Add properties to instance data
                for (key, value) in properties
                    values = get!(instance_data, key, typeof(value)[])
                    push!(values, value)
                end

                # Read metrics
                metrics = CSV.read(output_file.metrics_file, DataFrame)
                append!(batch_metrics, metrics)

                next!(prog)
            catch e
                @warn "Error processing file for summary" file = output_file.metrics_file exception = e
            end
        end

        # Append batch metrics to all metrics
        append!(all_metrics, batch_metrics)
    end

    # Convert instance data to DataFrame and combine
    df = hcat(DataFrame(instance_data), all_metrics)

    # Write summary file
    summary_file = joinpath(output_dir, method.name, method.version, "summary.csv")

    try
        CSV.write(summary_file, df)
    catch e
        @error "Failed to write summary file" summary_file exception = e
        return ""
    end

    return summary_file
end

"""
    extract_filename_properties(metrics_file::String) -> Dict{String,Any}

Extract properties from a metrics filename more efficiently.
"""
function extract_filename_properties(metrics_file::String)::Dict{String,Any}
    properties = Dict{String,Any}()

    basename_parts = split(basename(metrics_file), "_")
    extract_properties = [("nNodes", Int), ("density", Int), ("sim", Int), ("run", Int)]

    for (property, type) in extract_properties
        property_index = findfirst(x -> startswith(x, property), basename_parts)
        if !isnothing(property_index)
            str_value = chopprefix(basename_parts[property_index], property)
            try
                value = parse(type, str_value)
                properties[property] = value
            catch
                # Skip if parsing fails
            end
        end
    end

    return properties
end

# Task processing functions ----------------------------------------------------

"""
    process_single_task(
        task::ComputeTask, 
        method::AbstractMethod, 
        output_dir::String,
        dir_paths::Dict{String,Dict{String,String}},
        force_recompute::Bool
    ) -> Union{OutputFiles,Nothing}

Process a single task and return its output files, or nothing if skipped.
"""
function process_single_task(
    task::ComputeTask,
    method::AbstractMethod,
    output_dir::String,
    dir_paths::Dict{String,Dict{String,String}},
    force_recompute::Bool
)::Union{OutputFiles,Nothing}
    # Check if we should process this task
    if !should_process_task(task, method, output_dir, dir_paths, force_recompute)
        return nothing
    end

    # Solve the task
    result = solve_task(task, method)

    # Ensure the directory for this graph type exists
    if !haskey(dir_paths, result.graph_type)
        graph_types = Set{String}([result.graph_type])
        new_paths = ensure_directories(output_dir, method, graph_types)
        merge!(dir_paths, new_paths)
    end

    # Write results to disk
    output_files = write_solution_files(result, output_dir, dir_paths)

    return output_files
end

"""
    process_task_by_index(
        task_index::Int, 
        dataset::AbstractDataset,
        method::AbstractMethod, 
        output_dir::String;
        num_runs::Int=5,
        force_recompute::Bool=false
    ) -> Union{OutputFiles,Nothing}

Process a specific task by its index and save the result to the output directory.
This is useful for distributed processing where each job only processes a subset of tasks.

Returns the OutputFiles for the processed task, or nothing if skipped.
"""
function process_task_by_index(
    task_index::Int,
    dataset::AbstractDataset,
    method::AbstractMethod,
    output_dir::String;
    num_runs::Int=5,
    force_recompute::Bool=false
)::Union{OutputFiles,Nothing}
    # Create all tasks
    all_instances = collect(Datasets.instances(dataset))

    if isempty(all_instances)
        error("No instances found in dataset")
    end

    # Calculate total tasks
    total_tasks = length(all_instances) * num_runs

    if task_index < 1 || task_index > total_tasks
        error("Task index out of range: $task_index (valid range: 1-$total_tasks)")
    end

    # Calculate which instance and run this corresponds to
    instance_idx = (task_index - 1) ÷ num_runs + 1
    run_idx = (task_index - 1) % num_runs + 1

    # Get the instance
    instance = all_instances[instance_idx]

    # Create the task
    task = ComputeTask(instance, run_idx)

    # Ensure directories exist
    dir_paths = ensure_directories(output_dir, method)

    # Process the task
    return process_single_task(task, method, output_dir, dir_paths, force_recompute)
end

"""
    process_task_range(
        start_index::Int, 
        end_index::Int, 
        dataset::AbstractDataset,
        method::AbstractMethod, 
        output_dir::String;
        num_runs::Int=5,
        force_recompute::Bool=false
    ) -> OutputCollection

Process a range of tasks and save the results.
Useful for batch processing a subset of tasks.

Returns the OutputFiles for the processed tasks.
"""
function process_task_range(
    start_index::Int,
    end_index::Int,
    dataset::AbstractDataset,
    method::AbstractMethod,
    output_dir::String;
    num_runs::Int=5,
    force_recompute::Bool=false
)::OutputCollection
    # Create all tasks
    all_instances = collect(Datasets.instances(dataset))

    if isempty(all_instances)
        error("No instances found in dataset")
    end

    # Calculate total tasks
    total_tasks = length(all_instances) * num_runs

    if start_index < 1 || start_index > total_tasks
        error("Start index out of range: $start_index (valid range: 1-$total_tasks)")
    end

    if end_index < start_index || end_index > total_tasks
        error("End index out of range: $end_index (valid range: $start_index-$total_tasks)")
    end

    # Create progress meter
    num_tasks = end_index - start_index + 1
    prog = Progress(num_tasks, desc="Processing tasks $start_index-$end_index:", dt=1.0)

    # Ensure directories exist for common graph types
    dir_paths = ensure_directories(output_dir, method)

    # Process each task in the range
    output_files = OutputCollection()

    for task_idx in start_index:end_index
        # Calculate which instance and run this corresponds to
        instance_idx = (task_idx - 1) ÷ num_runs + 1
        run_idx = (task_idx - 1) % num_runs + 1

        # Get the instance
        instance = all_instances[instance_idx]

        # Create the task
        task = ComputeTask(instance, run_idx)

        # Process the task
        result_files = process_single_task(task, method, output_dir, dir_paths, force_recompute)

        # Add to output if not skipped
        if !isnothing(result_files)
            push!(output_files, result_files)
        end

        # Update progress
        next!(prog)
    end

    return output_files
end

# Main processing function -----------------------------------------------------

"""
    process_dataset(
        dataset::AbstractDataset, 
        method::AbstractMethod, 
        output_dir::String; 
        num_runs::Int=5, 
        force_recompute::Bool=false,
        task_range::Union{Nothing,Tuple{Int,Int}}=nothing,
        batch_size::Int=100
    ) -> OutputCollection

Process all instances in a dataset using the specified method and save results.
Sequential implementation for simplicity and reliability.

# Arguments
- `dataset`: The dataset containing instances to process
- `method`: The method to use for solving instances
- `output_dir`: Directory where results will be saved
- `num_runs`: Number of solutions to compute for each instance
- `force_recompute`: Whether to recompute solutions that already exist
- `task_range`: Optional range of tasks to process (for distributed execution)
- `batch_size`: Number of tasks to process in a single batch (for memory management)

# Returns
- Collection of output file paths for all written solutions
"""
function process_dataset(
    dataset::AbstractDataset,
    method::AbstractMethod,
    output_dir::String;
    num_runs::Int=5,
    force_recompute::Bool=false,
    task_range::Union{Nothing,Tuple{Int,Int}}=nothing,
    batch_size::Int=100
)::OutputCollection
    # 1. Setup phase
    # Ensure output directory exists
    mkpath(output_dir)

    # Collect instances
    all_instances = collect(Datasets.instances(dataset))

    # Early return if no instances to process
    if isempty(all_instances)
        @info "No instances found in the dataset."
        return OutputCollection()
    end

    # Total number of tasks
    total_tasks = length(all_instances) * num_runs

    # Process a specific range if requested
    if !isnothing(task_range)
        start_idx, end_idx = task_range
        return process_task_range(
            start_idx, end_idx, dataset, method, output_dir,
            num_runs=num_runs, force_recompute=force_recompute
        )
    end

    # Create all tasks
    all_tasks = [ComputeTask(instance, run) for instance in all_instances for run in 1:num_runs]

    # Create directory structure for all known graph types
    graph_types = Set{String}()
    for instance in all_instances
        if hasproperty(instance, :metadata) &&
           !isnothing(instance.metadata) &&
           haskey(instance.metadata, :graph_type)
            push!(graph_types, string(instance.metadata[:graph_type]))
        end
    end
    dir_paths = ensure_directories(output_dir, method, graph_types)

    # 2. Process tasks in batches to limit memory usage
    output_files = OutputCollection()
    num_batches = ceil(Int, length(all_tasks) / batch_size)

    prog = Progress(length(all_tasks), desc="Processing dataset:", dt=1.0)

    for batch_idx in 1:num_batches
        start_idx = (batch_idx - 1) * batch_size + 1
        end_idx = min(batch_idx * batch_size, length(all_tasks))

        # Process this batch
        for task_idx in start_idx:end_idx
            task = all_tasks[task_idx]

            # Process task and check if it wasn't skipped
            result_files = process_single_task(task, method, output_dir, dir_paths, force_recompute)
            if !isnothing(result_files)
                push!(output_files, result_files)
            end

            next!(prog)
        end
    end

    # 3. Generate summary file if there are results
    if !isempty(output_files)
        println("\nGenerating summary file...")
        create_summary_files(output_files, method, output_dir)
    end

    return output_files
end

# Find all results in a directory (useful for creating summaries) -------------

"""
    find_all_output_files(output_dir::String, method::AbstractMethod) -> OutputCollection
    
Find all output files for a given method in the output directory.
Useful for regenerating summaries or gathering results from separate runs.
"""
function find_all_output_files(output_dir::String, method::AbstractMethod)::OutputCollection
    base_dir = joinpath(output_dir, method.name, method.version)

    if !isdir(base_dir)
        @warn "No results directory found at $base_dir"
        return OutputCollection()
    end

    output_files = OutputCollection()

    # Iterate through all graph type directories
    for graph_type in readdir(base_dir)
        graph_dir = joinpath(base_dir, graph_type)

        if !isdir(graph_dir)
            continue
        end

        perm_dir = joinpath(graph_dir, "Permutations")
        metrics_dir = joinpath(graph_dir, "Metrics")

        if !isdir(perm_dir) || !isdir(metrics_dir)
            continue
        end

        # Find matching permutation and metrics files
        for perm_file in readdir(perm_dir)
            if !endswith(perm_file, ".csv")
                continue
            end

            # Extract base name for matching
            base_name = replace(perm_file, r"\.csv$" => "")
            metrics_file = joinpath(metrics_dir, "$(base_name)_metrics.csv")

            # Only add if both files exist
            if isfile(joinpath(perm_dir, perm_file)) && isfile(metrics_file)
                push!(output_files, OutputFiles(
                    joinpath(perm_dir, perm_file),
                    metrics_file
                ))
            end
        end
    end

    return output_files
end

# Export public interface
export process_dataset, process_task_by_index, process_task_range,
    find_all_output_files, create_summary_files,
    ComputeTask, SolutionResult, OutputFiles
