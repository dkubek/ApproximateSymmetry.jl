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
        unload!(instance)
        return nothing
end

function solve_task(
        instance::AbstractInstance,
        method::AbstractMethod,
        ::Type{T}=Float64;
        nruns=1
) where {T<:Real}
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

        buffer = Vector{Int}(undef, n)

        open(filename, "w") do io
                # Write CSV line
                for i in 1:n
                        # Extract this row across all solutions
                        for (sol_idx, solution) in enumerate(solutions)
                                P = get_result(solution)
                                buffer[sol_idx] = findfirst(x -> x == 1, view(P, i, :)) - 1
                        end

                        line = join(buffer, ',') * '\n'
                        write(io, line)
                end
        end

        return filename
end

function write_metrics(
        solutions::Vector{Solution{T}},
        filename::String
) where {T<:Number}
        n = length(solutions)
        column_data = Dict{Symbol,Vector}()

        # Collect all possible keys
        all_keys = Set{Symbol}()
        for solution in solutions
                union!(all_keys, keys(get_metrics(solution)))
        end

        # Convert to sorted array for consistent ordering
        sorted_keys = sort!(collect(all_keys))

        # Pre-allocate column vectors
        for key in all_keys
                column_data[key] = Vector{Any}(undef, n)
        end

        # Preallocate a single buffer for string construction
        # Use IOBuffer instead of string concatenation
        io_buffer = IOBuffer()

        # # Fill the columns
        # for (i, solution) in enumerate(solutions)
        #         metrics = get_metrics(solution)
        #         for key in all_keys
        #                 column_data[key][i] = get(metrics, key, missing)
        #         end
        # end

        # # Create DataFrame directly from columns (more efficient)
        # open(filename, "w") do io
        #         header = join(String.(collect(all_keys)), ",") * "\n"
        #         write(io, header)

        #         for i in 1:n
        #                 row = join([string(column_data[key][i]) for key in all_keys], ",") * "\n"
        #                 write(io, row)
        #         end
        # end

        open(filename, "w") do io
                # Write header only once
                for (i, key) in enumerate(sorted_keys)
                        i > 1 && write(io_buffer, ',')
                        write(io_buffer, string(key))
                end
                write(io_buffer, '\n')
                write(io, String(take!(io_buffer)))

                # Write rows
                for solution in solutions
                        metrics = get_metrics(solution)
                        for (i, key) in enumerate(sorted_keys)
                                i > 1 && write(io_buffer, ',')
                                value = get(metrics, key, missing)
                                write(io_buffer, string(value))
                        end
                        write(io_buffer, '\n')
                        write(io, String(take!(io_buffer)))
                end
        end



        return filename
end
