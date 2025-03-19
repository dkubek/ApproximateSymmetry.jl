module ApproximateSymmetry

using LinearAlgebra
using NPZ
using CSV
using DataFrames
using ProgressMeter
using Dates

# Export key types and functions
export
    # Abstract types
    AbstractInstance,
    AbstractDataset,
    AbstractMethod,
    AbstractMetric,
    AbstractOutputFormat,

    # Concrete instance types
    MatrixInstance,

    # Dataset types
    NPZDataset,

    # Loading functions
    load_instance,

    # Method interfaces
    solve,

    # Metrics
    compute_metric,
    S,

    # Output formatting
    save_results

# Include core components
include("abstractions.jl")
include("instances.jl")
include("datasets/pidnebesna.jl")
include("methods/base.jl")
include("io/readers/npz_loader.jl")
include("io/writers/csv.jl")

"""
    process_dataset(dataset::NPZDataset, method::AbstractMethod, output_dir::String; 
                    num_runs=5, force_recompute=false)

Process all instances in a dataset using the specified method and save results to the output directory.
"""
function process_dataset(dataset::NPZDataset, method::AbstractMethod, output_dir::String;
    num_runs=5, force_recompute=false)
    # Create output directories
    method_dir = joinpath(output_dir, method.name)
    version_dir = joinpath(method_dir, method.version)
    model_output_dir = joinpath(version_dir, dataset.graph_type)
    permutations_dir = joinpath(model_output_dir, "Permutations")
    timing_dir = joinpath(model_output_dir, "Timing")
    s_metric_dir = joinpath(model_output_dir, "S_Metric")

    # Create directories
    mkpath(permutations_dir)
    mkpath(timing_dir)
    mkpath(s_metric_dir)

    # Output formatters
    csv_formatter = CSVOutputFormat()

    # Metrics
    s_metric = SMetric()
    time_metric = ExecutionTimeMetric()

    # Process each instance
    progress = Progress(length(collect(iterate_instances(dataset))), 1, "Processing instances...")

    for (base_name, sim_idx, instance) in iterate_instances(dataset)
        # Check if results already exist
        perm_file = joinpath(permutations_dir, "$(base_name)_sim$(sim_idx).csv")
        timing_file = joinpath(timing_dir, "$(base_name)_sim$(sim_idx)_timing.csv")
        s_file = joinpath(s_metric_dir, "$(base_name)_sim$(sim_idx)_s_metric.csv")

        # Skip if all results exist and not forcing recomputation
        if !force_recompute && isfile(perm_file) && isfile(timing_file) && isfile(s_file)
            next!(progress)
            continue
        end

        # Run the method multiple times
        permutations = Vector{Matrix}()
        times = Vector{Float64}()
        s_values = Vector{Float64}()

        for _ in 1:num_runs
            # Solve the instance
            solution, execution_time = solve(method, instance)

            # Calculate metrics
            s_value = compute_metric(s_metric, instance, solution)

            # Store results
            push!(permutations, solution)
            push!(times, execution_time)
            push!(s_values, s_value)
        end

        # Save permutation matrices
        save_permutations(csv_formatter, permutations, perm_file)

        # Save timing data
        save_metrics(csv_formatter, times, fill("time", length(times)), timing_file)

        # Save S metric data
        save_metrics(csv_formatter, s_values, fill("s_metric", length(s_values)), s_file)

        next!(progress)
    end

    # Compile and save timing summary
    time_summary_file = joinpath(model_output_dir, "$(method.name)_time_$(dataset.graph_type).csv")
    compile_timing_summary(timing_dir, time_summary_file, num_runs)

    # Compile and save S metric summary
    s_summary_file = joinpath(model_output_dir, "$(method.name)_s_metric_$(dataset.graph_type).csv")
    compile_s_metric_summary(s_metric_dir, s_summary_file, num_runs)
end

"""
    compile_timing_summary(timing_dir::String, output_file::String, num_runs::Int)

Compile all individual timing files into a summary CSV.
"""
function compile_timing_summary(timing_dir::String, output_file::String, num_runs::Int)
    # Base data structure for results
    results = Dict{String,Dict{Int,Vector{Float64}}}()

    # Process all CSV files in the timing directory
    for file in readdir(timing_dir)
        if endswith(file, "_timing.csv")
            # Extract base_name and sim_idx from filename
            match_result = match(r"(.+)_sim(\d+)_timing\.csv", file)
            if match_result !== nothing
                base_name = match_result.captures[1]
                sim_idx = parse(Int, match_result.captures[2])

                # Initialize if needed
                if !haskey(results, base_name)
                    results[base_name] = Dict{Int,Vector{Float64}}()
                end

                # Read the timing data
                df = CSV.read(joinpath(timing_dir, file), DataFrame)

                # Store the timing data
                results[base_name][sim_idx] = df.metric
            end
        end
    end

    # Create summary DataFrame
    summary_cols = ["base_name", "simulation"]
    run_cols = ["run$i" for i in 1:num_runs]
    append!(summary_cols, run_cols)

    summary_df = DataFrame([name => [] for name in summary_cols])

    # Fill summary DataFrame
    for base_name in sort(collect(keys(results)))
        for sim_idx in sort(collect(keys(results[base_name])))
            row = Dict("base_name" => base_name, "simulation" => sim_idx)

            times = results[base_name][sim_idx]
            for i in 1:num_runs
                row["run$i"] = i <= length(times) ? times[i] : NaN
            end

            push!(summary_df, row)
        end
    end

    # Save summary
    save_summary(CSVOutputFormat(), summary_df, output_file)
end

"""
    compile_s_metric_summary(s_metric_dir::String, output_file::String, num_runs::Int)

Compile all individual S metric files into a summary CSV.
"""
function compile_s_metric_summary(s_metric_dir::String, output_file::String, num_runs::Int)
    # Base data structure for results
    results = Dict{String,Dict{Int,Vector{Float64}}}()

    # Process all CSV files in the S metric directory
    for file in readdir(s_metric_dir)
        if endswith(file, "_s_metric.csv")
            # Extract base_name and sim_idx from filename
            match_result = match(r"(.+)_sim(\d+)_s_metric\.csv", file)
            if match_result !== nothing
                base_name = match_result.captures[1]
                sim_idx = parse(Int, match_result.captures[2])

                # Initialize if needed
                if !haskey(results, base_name)
                    results[base_name] = Dict{Int,Vector{Float64}}()
                end

                # Read the S metric data
                df = CSV.read(joinpath(s_metric_dir, file), DataFrame)

                # Store the S metric data
                results[base_name][sim_idx] = df.metric
            end
        end
    end

    # Create summary DataFrame
    summary_cols = ["base_name", "simulation"]
    run_cols = ["run$i" for i in 1:num_runs]
    append!(summary_cols, run_cols)

    summary_df = DataFrame([name => [] for name in summary_cols])

    # Fill summary DataFrame
    for base_name in sort(collect(keys(results)))
        for sim_idx in sort(collect(keys(results[base_name])))
            row = Dict("base_name" => base_name, "simulation" => sim_idx)

            s_values = results[base_name][sim_idx]
            for i in 1:num_runs
                row["run$i"] = i <= length(s_values) ? s_values[i] : NaN
            end

            push!(summary_df, row)
        end
    end

    # Save summary
    save_summary(CSVOutputFormat(), summary_df, output_file)
end

end
