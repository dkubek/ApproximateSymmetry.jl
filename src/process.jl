using ApproximateSymmetry
using DelimitedFiles
using CSV
using DataFrames

# Type definitions --------------------------------------------------------------

struct OutputFiles
        permutation_file::String
        metrics_file::String
end

function process(
        instance::AbstractInstance,
        method::AbstractMethod,
        result_dir::String,
        T=Float64;
        nruns=1
)
        # @info "Processing" instance.id
        solutions = solve_task(instance, method, T; nruns=nruns)
        write_solution_files(instance, method, solutions, result_dir)
        return nothing
end

function solve_task(
        instance::AbstractInstance,
        method::AbstractMethod,
        ::Type{T}=Float64;
        nruns=1
)::Vector{Solution{T}} where {T<:Number}

        solutions = Vector{Solution{T}}(undef, nruns)
        for i in eachindex(solutions)
                solutions[i] = Methods.solve(method, instance)
        end
        solutions
end

function task_already_processed(
        instance::AbstractInstance,
        method::AbstractMethod,
        output_dir::String,
)::Bool
        # TODO: Debug log checking if task exists
        output_files = get_output_filenames(instance, method, output_dir)
        return result_exists(output_files)
end

function get_output_filenames(
        instance::AbstractInstance,
        method::AbstractMethod,
        output_dir::String,
)::OutputFiles
        graph_type = get(instance.metadata, :graph_type, "unknown_graph_type")

        permutation_file = joinpath(
                output_dir,
                method.name,
                method.version,
                graph_type,
                "permutations",
                "$(instance.id).csv"
        )
        metrics_file = joinpath(
                output_dir,
                method.name,
                method.version,
                graph_type,
                "metrics",
                "$(instance.id).csv"
        )

        OutputFiles(permutation_file, metrics_file)
end

function result_exists(output_files::OutputFiles)::Bool
        # TODO: Debug log checking if output files exists, also print output files into debug output
        return isfile(output_files.permutation_file) && isfile(output_files.metrics_file)
end

## File IO functions ------------------------------------------------------------

function write_solution_files(
        instance::AbstractInstance,
        method::AbstractMethod,
        solutions::Vector{Solution{T}},
        result_dir::String,
)::OutputFiles where {T<:Number}
        # Get filenames
        output_files = get_output_filenames(instance, method, result_dir)
        ensure_directories(output_files)

        try
                write_permutation(solutions, output_files.permutation_file)
                write_metrics(solutions, output_files.metrics_file)
        catch e
                @error "Failed to write solution files" instance.id exception = e
                isfile(output_files.permutation_file) && rm(output_files.permutation_file, force=true)
                isfile(output_files.metrics_file) && rm(output_files.metrics_file, force=true)
                rethrow(e)
        end

        return output_files
end


function ensure_directories(output_files::OutputFiles)
        mkpath(dirname(output_files.permutation_file))
        mkpath(dirname(output_files.metrics_file))
end


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


function write_metrics(
        solutions::Vector{Solution{T}},
        filename::String
)::String where {T<:Number}
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
