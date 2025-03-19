"""
    process_dataset(dataset::AbstractDataset, method::AbstractMethod, output_dir::String, format::AbstractOutputFormat; 
                    num_runs=5, force_recompute=false)

Process all instances in a dataset using the specified method and save results to the output directory.
Uses the specified output format for writing results.
"""
function process_dataset(dataset::AbstractDataset, method::AbstractMethod, output_dir::String, format::AbstractOutputFormat;
    num_runs=5, force_recompute=false)
    error("process_dataset not implemented for dataset type $(typeof(dataset))")
end

"""
    process_dataset(dataset::NPZDataset, method::AbstractMethod, output_dir::String, format::AbstractOutputFormat; 
                    num_runs=5, force_recompute=false)

Process all instances in an NPZ dataset using the specified method and save results.
"""
function process_dataset(dataset::NPZDataset, method::AbstractMethod, output_dir::String, format::AbstractOutputFormat;
    num_runs=5, force_recompute=false)

    # Create output directories
    method_dir = joinpath(output_dir, method.name)
    version_dir = joinpath(method_dir, method.version)
    model_output_dir = joinpath(version_dir, dataset.graph_type)
    permutations_dir = joinpath(model_output_dir, "Permutations")
    metrics_dir = joinpath(model_output_dir, "Metrics")

    # Create directories
    mkpath(permutations_dir)
    mkpath(metrics_dir)

    # Process each instance
    progress = Progress(length(collect(iterate_instances(dataset))), 1, "Processing instances...")

    for instance in iterate_instances(dataset)
        base_name = instance.metadata[:base_name]
        sim_idx = instance.metadata[:simulation]

        # Output file paths
        perm_file = joinpath(permutations_dir, "$(base_name)_sim$(sim_idx).csv")
        metrics_file = joinpath(metrics_dir, "$(base_name)_sim$(sim_idx)_metrics.csv")

        # Skip if results already exist and not forcing recomputation
        if !force_recompute && isfile(perm_file) && isfile(metrics_file)
            next!(progress)
            continue
        end

        # Run the method multiple times
        solutions = Vector{Solution}()

        for _ in 1:num_runs
            # Solve the instance
            solution = solve(method, instance)
            push!(solutions, solution)
        end

        # Save each solution separately
        for (run_idx, solution) in enumerate(solutions)
            run_perm_file = joinpath(permutations_dir, "$(base_name)_sim$(sim_idx)_run$(run_idx).csv")
            write_solution(solution, format, run_perm_file)
        end

        # Save the combined permutation results (for backward compatibility)
        perm_matrices = [get_result(sol) for sol in solutions]
        combined_solution = Solution(perm_matrices)
        write_solution(combined_solution, format, perm_file)

        # Save metrics
        # For each supported metric, create a summary
        for metric_name in supported_metrics(method)
            metric_values = [get_metric(sol, metric_name, NaN) for sol in solutions]
            metric_file = joinpath(metrics_dir, "$(base_name)_sim$(sim_idx)_$(metric_name).csv")

            # Create a solution with just this metric
            metric_solution = Solution(missing, Dict{Symbol,Any}(metric_name => metric_values))
            write_solution(metric_solution, format, metric_file)
        end

        next!(progress)
    end

    # Compile and save summaries for each metric
    for metric_name in supported_metrics(method)
        summary_file = joinpath(model_output_dir, "$(method.name)_$(metric_name)_$(dataset.graph_type).csv")
        write_summary(format, metrics_dir, summary_file; metric=metric_name, num_runs=num_runs)
    end

    return nothing
end

# Export process_dataset
export process_dataset
