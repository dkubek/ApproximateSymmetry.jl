using DelimitedFiles

"""
Output formatter for saving results to CSV files.
"""
struct CSVOutputFormat <: AbstractOutputFormat
end

"""
Write a solution to the specified path using CSV format.
"""
function write_solution(::CSVOutputFormat, solution::Solution, outdir::String, basename::String)
        # Create the directory if it doesn't exist
        mkpath(dirname(outdir))

        # Get the actual result
        result = get_result(solution)

        if isa(result, Matrix)
                result = vec([Matrix])
        end

        result = result |> collect
        println(result)
        if isa(result, Vector) && all(x -> isa(x, Matrix), filter(!isnothing, result))
                permutations = filter(!isnothing, result)
                no_runs = size(permutations, 1)
                @assert no_runs > 0
                n = size(first(permutations), 1)

                sigmas = vec()
                for (i, P) in enumerate(result)
                        sigma = [findfirst(x -> x == 1, P[i, :]) for i in 1:n]
                        push!(sigmas, sigma)
                end

                data = hcat(sigmas)

                csv_file = basename * ".csv"
                headers = ["run$i" for i in 1:no_runs]
                open(csv_file, "w", encoding="UTF-8") do io
                        writedlm(io, [headers'; data], ',')
                end
        end

        # If there are metrics, save them to a separate file
        # metrics = get_metrics(solution)
        # if !isempty(metrics)
        #         metrics_path = replace(path, r"\.csv$" => "_metrics.csv")

        #         # Convert any nothing values to missing
        #         metrics_fixed = Dict{Symbol,Any}()
        #         for (k, v) in metrics
        #                 metrics_fixed[k] = v === nothing ? missing : v
        #         end

        #         metrics_df = DataFrame(metric=collect(keys(metrics_fixed)), value=collect(values(metrics_fixed)))
        #         CSV.write(metrics_path, metrics_df)
        # end

        return nothing
end

"""
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
