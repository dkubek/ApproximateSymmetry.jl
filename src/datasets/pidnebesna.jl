"""
    NPZDataset <: AbstractDataset

A dataset implementation for NPZ files containing matrices.

# Fields
- `path::String`: Path to the dataset directory
- `graph_type::String`: Type of graph represented in this dataset
- `instances::Dict{String, Vector{MatrixInstance}}`: Cached instances by base name
- `metadata::Dict{Symbol, Any}`: Additional dataset metadata
"""
struct NPZDataset <: AbstractDataset
    path::String
    graph_type::String
    instances::Dict{String,Vector{MatrixInstance}}
    metadata::Dict{Symbol,Any}
end

"""
    NPZDataset(path::String, graph_type::String)

Construct a new NPZ dataset from the specified path and graph type.
"""
function NPZDataset(path::String, graph_type::String)
    NPZDataset(path, graph_type, Dict{String,Vector{MatrixInstance}}(), Dict{Symbol,Any}())
end

"""
    Base.getindex(dataset::NPZDataset, base_name::String)

Get all instances for a particular base name.
"""
function Base.getindex(dataset::NPZDataset, base_name::String)
    if !haskey(dataset.instances, base_name)
        # Lazy-load instances when first requested
        dataset.instances[base_name] = _load_base_instances(dataset, base_name)
    end
    return dataset.instances[base_name]
end

"""
    Base.getindex(dataset::NPZDataset, base_name::String, sim_idx::Int)

Get a specific simulation instance.
"""
function Base.getindex(dataset::NPZDataset, base_name::String, sim_idx::Int)
    instances = dataset[base_name]
    return instances[sim_idx]
end

"""
    _load_base_instances(dataset::NPZDataset, base_name::String)

Internal method to load all simulation instances for a base name.
"""
function _load_base_instances(dataset::NPZDataset, base_name::String)
    # Find the file path for this base name
    file_path = joinpath(dataset.path, dataset.graph_type, "$(base_name).npz")

    if !isfile(file_path)
        error("NPZ file not found: $file_path")
    end

    # Load all simulation matrices from the NPZ file
    data = npzread(file_path)

    # Create instances for each simulation
    instances = Vector{MatrixInstance}()

    for sim_idx = 0:38  # Assuming 39 simulations per file
        if haskey(data, "$sim_idx")
            matrix = data["$sim_idx"]
            id = "$(base_name)_sim$(sim_idx)"
            instance = MatrixInstance(id, matrix)

            # Add metadata
            set_metadata!(instance, :graph_type, dataset.graph_type)
            set_metadata!(instance, :base_name, base_name)
            set_metadata!(instance, :simulation, sim_idx)

            push!(instances, instance)
        end
    end

    return instances
end

"""
    load_dataset(path::String; recursive=true)

Load all datasets from a directory structure.
Returns a dictionary of NPZDataset objects keyed by graph type.
"""
function load_dataset(path::String; recursive=true)
    datasets = Dict{String,NPZDataset}()

    if !isdir(path)
        error("Dataset directory does not exist: $path")
    end

    # Find all graph type directories
    dir_entries = readdir(path, join=true)
    graph_dirs = filter(isdir, dir_entries)

    # Create a dataset for each graph type
    for graph_dir in graph_dirs
        graph_type = basename(graph_dir)
        datasets[graph_type] = NPZDataset(path, graph_type)
    end

    return datasets
end

"""
    iterate_instances(dataset::NPZDataset)

Iterate through all instances in the dataset.
Returns tuples of (base_name, sim_idx, instance).
"""
function iterate_instances(dataset::NPZDataset)
    # Get all NPZ files in the directory
    files = filter(
        f -> isfile(f) && endswith(f, ".npz") && !contains(f, "_allInfo"),
        readdir(joinpath(dataset.path, dataset.graph_type), join=true),
    )

    # Process each file and yield instances
    Channel() do channel
        for file_path in files
            base_name = replace(basename(file_path), r"\.npz$" => "")
            instances = dataset[base_name]

            for (sim_idx, instance) in enumerate(instances)
                put!(channel, instance)
            end
        end
    end
end
