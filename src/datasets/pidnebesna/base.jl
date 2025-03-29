using DelimitedFiles
using CSV

"""
A dataset implementation for NPZ files containing matrices.
"""
struct NPZDataset <: AbstractDataset
    path::String
    instances::Dict{String,Vector{MultipleSimulationInstance}}
end

"""
Construct a new NPZ dataset from the specified path and graph type.
"""
function NPZDataset(path::String)
    NPZDataset(path, Dict{String,Vector{MultipleSimulationInstance}}())
end

"""
    load_dataset(path::String; recursive=true)

Load all datasets from a directory structure.
Returns a dictionary of NPZDataset objects keyed by graph type.
"""
function load_dataset(::Type{NPZDataset}, path::String; recursive=true)
    dataset = NPZDataset(path)

    if !isdir(path)
        error("Dataset directory does not exist: $path")
    end

    # Find all graph type directories
    dir_entries = readdir(path, join=true)
    graph_dirs = filter(isdir, dir_entries)

    # Create a dataset for each graph type
    for graph_dir in graph_dirs
        graph_type = basename(graph_dir)

        dataset.instances[graph_type] = Vector{MultipleSimulationInstance}()
        for instance_filename in find_instance_files(graph_dir)
            instance_id = replace(basename(instance_filename), r"\.npz$" => "")
            simulations = load_npz_instance(instance_filename, instance_id)
            push!(
                dataset.instances[graph_type],
                MultipleSimulationInstance(
                    instance_id,
                    simulations,
                    instance_filename,
                )
            )
        end
    end

    return dataset
end

# Get all NPZ files in the directory
find_instance_files(path::String) =
    filter(
        f -> isfile(f) && endswith(f, ".npz") && !contains(f, "_allInfo"),
        readdir(path, join=true),
    )


"""
Internal method to load all simulation instances for a base name.
"""
function load_npz_instance(file_path::String, base_name::String)
    if !isfile(file_path)
        error("NPZ file not found: $file_path")
    end

    # Load all simulation matrices from the NPZ file
    data = npzread(file_path)

    # Create instances for each simulation
    simulations = Vector{MatrixInstance}()

    for sim_idx = 0:38  # Assuming 39 simulations per file
        matrix = data["$sim_idx"]
        id = "$(base_name)_sim$(sim_idx)"

        n = size(matrix, 1)
        simulation = MatrixInstance(id, matrix, n, file_path)

        push!(simulations, simulation)
    end

    return simulations
end

# """
# Create a summary CSV file by compiling metric results from CSV files in the directory.
# """
# function write_summary(directory::String, output_file::String;)
#         metric::Symbol, num_runs::Int=5)
#         # Base data structure for results
#         results = Dict{String,Dict{Int,Vector{Float64}}}()
# 
#         # Regex pattern for metric files
#         pattern = Regex("(.+)_sim(\\d+)_metrics\\.csv\$")
# 
#         # Process all CSV files in the directory
#         for file in readdir(directory, join=true)
#                 if occursin(pattern, basename(file))
#                         # Extract base_name and sim_idx from filename
#                         match_result = match(pattern, basename(file))
#                         if match_result !== nothing
#                                 base_name = match_result.captures[1]
#                                 sim_idx = parse(Int, match_result.captures[2])
# 
#                                 # Initialize if needed
#                                 if !haskey(results, base_name)
#                                         results[base_name] = Dict{Int,Vector{Float64}}()
#                                 end
# 
#                                 # Read the metrics data
#                                 df = CSV.read(file, DataFrame)
# 
#                                 # Check if this metric exists in the file
#                                 if any(df.metric .== metric)
#                                         # Get the first row where metric matches
#                                         metric_row = findfirst(df.metric .== metric)
#                                         metric_value = df[metric_row, :value]
# 
#                                         # Skip missing values
#                                         if !ismissing(metric_value)
#                                                 # Store the metric value
#                                                 if !haskey(results[base_name], sim_idx)
#                                                         results[base_name][sim_idx] = Float64[]
#                                                 end
#                                                 push!(results[base_name][sim_idx], convert(Float64, metric_value))
#                                         end
#                                 end
#                         end
#                 end
#         end
# 
#         # Create summary DataFrame
#         summary_cols = ["base_name", "simulation"]
#         run_cols = ["run$i" for i in 1:num_runs]
#         append!(summary_cols, run_cols)
# 
#         summary_df = DataFrame([name => [] for name in summary_cols])
# 
#         # Fill summary DataFrame
#         for base_name in sort(collect(keys(results)))
#                 for sim_idx in sort(collect(keys(results[base_name])))
#                         row = Dict("base_name" => base_name, "simulation" => sim_idx)
# 
#                         values = results[base_name][sim_idx]
#                         for i in 1:num_runs
#                                 row["run$i"] = i <= length(values) ? values[i] : missing
#                         end
# 
#                         push!(summary_df, row)
#                 end
#         end
# 
#         # Save summary
#         mkpath(dirname(output_file))
#         CSV.write(output_file, summary_df)
# 
#         return output_file
#     end
# end


export NPZDataset, load_dataset