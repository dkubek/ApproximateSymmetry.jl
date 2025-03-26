"""
Read a solution from a CSV file created with CSVOutputFormat.
"""
function read_solution(path::String, ::CSVOutputFormat, ::Type{Solution}; kwargs...)
    if !isfile(path)
        error("Solution file not found: $path")
    end

    # Read the CSV file
    df = CSV.read(path, DataFrame)

    # Check if this is a permutation matrix
    if "node" in names(df) && "destination" in names(df)
        n = nrow(df)
        perm_matrix = zeros(n, n)

        for i in 1:n
            j = df.destination[i]
            perm_matrix[i, j] = 1
        end

        # Read metrics if they exist
        metrics_path = replace(path, r"\.csv$" => "_metrics.csv")
        metrics = Dict{Symbol,Any}()

        if isfile(metrics_path)
            metrics_df = CSV.read(metrics_path, DataFrame)
            for row in eachrow(metrics_df)
                metrics[Symbol(row.metric)] = row.value
            end
        end

        return Solution(perm_matrix, metrics)
    else
        # This is just a metrics file
        metrics = Dict{Symbol,Any}()
        for row in eachrow(df)
            metrics[Symbol(row.metric)] = row.value
        end

        return Solution(nothing, metrics)
    end
end
