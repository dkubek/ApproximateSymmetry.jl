"""
    CSVOutputFormat <: AbstractOutputFormat

Output formatter for saving results to CSV files.
"""
struct CSVOutputFormat <: AbstractOutputFormat
end

"""
    write_solution(solution::Solution, format::CSVOutputFormat, path::String)

Write a solution to the specified path using CSV format.
"""
function write_solution(solution::Solution, ::CSVOutputFormat, path::String)
        # Create the directory if it doesn't exist
        mkpath(dirname(path))

        # Get the actual result
        result = get_result(solution)

        if isa(result, Matrix)
                # Handle matrix results (like permutation matrices)
                n = size(result, 1)
                # Convert permutation matrix to destination indices
                destinations = [findfirst(x -> x == 1, result[i, :]) for i in 1:n]

                # Create DataFrame and save to CSV
                df = DataFrame(node=1:n, destination=destinations)
                CSV.write(path, df)
        elseif isa(result, Vector) && all(x -> isa(x, Matrix), filter(!isnothing, result))
                # Handle a vector of matrices (like multiple permutations)
                # Get the first non-nothing matrix to determine size
                first_matrix = first(filter(!isnothing, result))
                n = size(first_matrix, 1)
                df = DataFrame(node=1:n)

                for (i, P) in enumerate(result)
                        if isnothing(P)
                                df[!, "run$i"] = fill(missing, n)
                        else
                                # Convert permutation matrix to destination indices
                                destinations = [findfirst(x -> x == 1, P[i, :]) for i in 1:n]
                                df[!, "run$i"] = destinations
                        end
                end

                CSV.write(path, df)
        else
                # For other result types, just save the metrics
                metrics = get_metrics(solution)
                if !isempty(metrics)
                        # Convert any nothing values to missing
                        metrics_fixed = Dict{Symbol,Any}()
                        for (k, v) in metrics
                                metrics_fixed[k] = v === nothing ? missing : v
                        end
                        df = DataFrame(metric=collect(keys(metrics_fixed)), value=collect(values(metrics_fixed)))
                        CSV.write(path, df)
                else
                        # Create an empty metrics file
                        df = DataFrame(metric=Symbol[], value=[])
                        CSV.write(path, df)
                end
        end

        # If there are metrics, save them to a separate file
        metrics = get_metrics(solution)
        if !isempty(metrics)
                metrics_path = replace(path, r"\.csv$" => "_metrics.csv")

                # Convert any nothing values to missing
                metrics_fixed = Dict{Symbol,Any}()
                for (k, v) in metrics
                        metrics_fixed[k] = v === nothing ? missing : v
                end

                metrics_df = DataFrame(metric=collect(keys(metrics_fixed)), value=collect(values(metrics_fixed)))
                CSV.write(metrics_path, metrics_df)
        end

        return path
end

"""
    write_summary(format::CSVOutputFormat, directory::String, output_file::String; 
                 metric::Symbol, num_runs::Int=5)

Create a summary CSV file by compiling metric results from CSV files in the directory.
"""
function write_summary(::CSVOutputFormat, directory::String, output_file::String;
        metric::Symbol, num_runs::Int=5)
        # Base data structure for results
        results = Dict{String,Dict{Int,Vector{Float64}}}()

        # Regex pattern for metric files
        pattern = Regex("(.+)_sim(\\d+)_metrics\\.csv\$")

        # Process all CSV files in the directory
        for file in readdir(directory, join=true)
                if occursin(pattern, basename(file))
                        # Extract base_name and sim_idx from filename
                        match_result = match(pattern, basename(file))
                        if match_result !== nothing
                                base_name = match_result.captures[1]
                                sim_idx = parse(Int, match_result.captures[2])

                                # Initialize if needed
                                if !haskey(results, base_name)
                                        results[base_name] = Dict{Int,Vector{Float64}}()
                                end

                                # Read the metrics data
                                df = CSV.read(file, DataFrame)

                                # Check if this metric exists in the file
                                if any(df.metric .== metric)
                                        # Get the first row where metric matches
                                        metric_row = findfirst(df.metric .== metric)
                                        metric_value = df[metric_row, :value]

                                        # Skip missing values
                                        if !ismissing(metric_value)
                                                # Store the metric value
                                                if !haskey(results[base_name], sim_idx)
                                                        results[base_name][sim_idx] = Float64[]
                                                end
                                                push!(results[base_name][sim_idx], convert(Float64, metric_value))
                                        end
                                end
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

                        values = results[base_name][sim_idx]
                        for i in 1:num_runs
                                row["run$i"] = i <= length(values) ? values[i] : missing
                        end

                        push!(summary_df, row)
                end
        end

        # Save summary
        mkpath(dirname(output_file))
        CSV.write(output_file, summary_df)

        return output_file
end
