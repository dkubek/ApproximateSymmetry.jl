"""
    save_results(formatter::AbstractOutputFormat, results, path::String)

Save results using a specific output format.
"""
function save_results end

"""
    CSVOutputFormat <: AbstractOutputFormat

Output formatter for saving results to CSV files.
"""
struct CSVOutputFormat <: AbstractOutputFormat
end

"""
    save_permutation(formatter::CSVOutputFormat, permutation::AbstractMatrix, path::String)

Save a permutation matrix to a CSV file.
"""
function save_permutation(::CSVOutputFormat, permutation::AbstractMatrix, path::String)
        # Create the directory if it doesn't exist
        mkpath(dirname(path))

        # Convert permutation matrix to destination indices
        n = size(permutation, 1)
        destinations = [findfirst(x -> x == 1, permutation[i, :]) for i in 1:n]

        # Create DataFrame and save to CSV
        df = DataFrame(node=1:n, destination=destinations)
        CSV.write(path, df)
end

"""
    save_permutations(formatter::CSVOutputFormat, permutations::Vector{<:AbstractMatrix}, path::String)

Save multiple permutation matrices to a CSV file with each as a column.
"""
function save_permutations(::CSVOutputFormat, permutations::Vector{<:AbstractMatrix}, path::String)
        # Create the directory if it doesn't exist
        mkpath(dirname(path))

        # Process each permutation
        n = size(permutations[1], 1)
        df = DataFrame(node=1:n)

        for (i, P) in enumerate(permutations)
                # Convert permutation matrix to destination indices
                destinations = [findfirst(x -> x == 1, P[i, :]) for i in 1:n]
                df[!, "run$i"] = destinations
        end

        # Save to CSV
        CSV.write(path, df)
end

"""
    save_metrics(formatter::CSVOutputFormat, metrics::Vector{<:Real}, labels::Vector{String}, path::String)

Save metrics to a CSV file.
"""
function save_metrics(::CSVOutputFormat, metrics::Vector{<:Real}, labels::Vector{String}, path::String)
        # Create the directory if it doesn't exist
        mkpath(dirname(path))

        # Create DataFrame and save to CSV
        df = DataFrame(run=1:length(metrics), metric=metrics)
        CSV.write(path, df)
end

"""
    save_summary(formatter::CSVOutputFormat, summary::DataFrame, path::String)

Save a summary DataFrame to a CSV file.
"""
function save_summary(::CSVOutputFormat, summary::DataFrame, path::String)
        # Create the directory if it doesn't exist
        mkpath(dirname(path))

        # Save to CSV
        CSV.write(path, summary)
end
