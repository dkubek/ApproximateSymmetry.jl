using NPZ

mutable struct PidnebesnaDataset <: AbstractDataset
    path::String
    graph_types::Vector{String}
    instances_map::Dict{String,Vector{MultipleSimulationInstance}}
    loaded::Bool
    metadata::Dict{Symbol,Any}
    
    function PidnebesnaDataset(path::String)
        if !isdir(path)
            throw(Common.ApproximateSymmetryError("Dataset directory does not exist: $path"))
        end
        
        new(path, String[], Dict{String,Vector{MultipleSimulationInstance}}(), false, Dict{Symbol,Any}())
    end
    
    function PidnebesnaDataset(path::String, graph_type::String)
        dataset = PidnebesnaDataset(path)
        load_graph_type!(dataset, graph_type)
        return dataset
    end
end

function load!(dataset::PidnebesnaDataset)
    if dataset.loaded
        return dataset
    end
    
    # Find all subdirectories (graph types)
    graph_type_dirs = filter(isdir, readdir(dataset.path, join=true))
    
    # Load each graph type
    for graph_type_dir in graph_type_dirs
        graph_type = basename(graph_type_dir)
        load_graph_type!(dataset, graph_type)
    end
    
    dataset.loaded = true
    return dataset
end

function load_graph_type!(dataset::PidnebesnaDataset, graph_type::String)
    graph_type_dir = joinpath(dataset.path, graph_type)
    
    if !isdir(graph_type_dir)
        throw(Common.ApproximateSymmetryError("Graph type directory does not exist: $graph_type_dir"))
    end
    
    # Add graph type if not already in the list
    if !(graph_type in dataset.graph_types)
        push!(dataset.graph_types, graph_type)
    end
    
    # Initialize instances vector for this graph type
    if !haskey(dataset.instances_map, graph_type)
        dataset.instances_map[graph_type] = Vector{MultipleSimulationInstance}()
    end
    
    # Find all NPZ files in the directory
    npz_files = filter(f -> isfile(f) && endswith(f, ".npz") && !contains(f, "_allInfo"), 
                       readdir(graph_type_dir, join=true))
    
    # Load each NPZ file as a MultipleSimulationInstance
    for npz_file in npz_files
        instance = load_npz_file(npz_file, graph_type)
        push!(dataset.instances_map[graph_type], instance)
    end
    
    return dataset
end

"""
    load_npz_file(file_path::String, graph_type::String)

Load an NPZ file into a MultipleSimulationInstance.
"""
function load_npz_file(file_path::String, graph_type::String)
    # Extract instance ID from filename
    base_name = basename(file_path)
    instance_id = replace(base_name, r"\.npz$" => "")
    
    # Load NPZ data
    data = NPZ.npzread(file_path)
    
    # Extract number of nodes if available in filename
    n_nodes = 0
    n_nodes_match = match(r"nNodes(\d+)", base_name)
    if n_nodes_match !== nothing
        n_nodes = parse(Int, n_nodes_match.captures[1])
    end
    
    # Create simulation instances
    simulations = Vector{Instances.MatrixInstance}()
    
    # NPZ files contain numbered simulations (0-based)
    for sim_idx in 0:38  # Assuming max 39 simulations
        sim_key = string(sim_idx)
        
        # Check if this simulation exists
        if !haskey(data, sim_key)
            break
        end
        
        # Get matrix for this simulation
        matrix = data[sim_key]
        
        # Verify matrix dimensions
        if n_nodes > 0 && size(matrix, 1) != n_nodes
            @warn "Matrix dimensions ($(size(matrix, 1))) do not match expected size ($n_nodes) for simulation $sim_idx in $base_name"
        end
        
        # Create simulation ID
        sim_id = "$(instance_id)_sim$(sim_idx)"
        
        # Create simulation instance
        simulation = MatrixInstance(sim_id, matrix)
        
        # Set metadata
        simulation.metadata[:graph_type] = graph_type
        simulation.metadata[:simulation_index] = sim_idx
        simulation.metadata[:base_instance] = instance_id
        simulation.metadata[:path] = file_path
        
        push!(simulations, simulation)
    end

    instance = MultipleSimulationInstance(instance_id, simulations)
    instance.metadata[:path] = file_path
    instance.metadata[:graph_type] = graph_type

    return instance
end

function Datasets.instances(dataset::PidnebesnaDataset)
    if !dataset.loaded
        load!(dataset)
    end
    
    # Collect all instances from all graph types
    all_instances = Iterators.flatten(values(dataset.instances_map))
    simulations = Iterators.flatten(map(instance -> instance.simulations, all_instances))
    return simulations
end

function instances(dataset::PidnebesnaDataset, graph_type::String)
    if !dataset.loaded
        load!(dataset)
    end
    
    if !haskey(dataset.instances_map, graph_type)
        throw(Common.ApproximateSymmetryError("Graph type not found in dataset: $graph_type"))
    end
    
    return dataset.instances_map[graph_type]
end

function Datasets.count_instances(dataset::PidnebesnaDataset, graph_type::String="")
    if !dataset.loaded
        load!(dataset)
    end
    
    if !isempty(graph_type)
        if !haskey(dataset.instances_map, graph_type)
            return 0
        end
        return length(dataset.instances_map[graph_type])
    else
        return sum(length(instances) for instances in values(dataset.instances_map))
    end
end

function get_graph_types(dataset::PidnebesnaDataset)
    if !dataset.loaded
        load!(dataset)
    end
    
    return dataset.graph_types
end

function filter_instances(dataset::PidnebesnaDataset, predicate::Function)
    # Create a new dataset
    filtered = PidnebesnaDataset(dataset.path)
    
    # Ensure original dataset is loaded
    if !dataset.loaded
        load!(dataset)
    end
    
    # Copy graph types
    filtered.graph_types = copy(dataset.graph_types)
    
    # Filter instances
    for graph_type in dataset.graph_types
        if !haskey(dataset.instances_map, graph_type)
            continue
        end
        
        # Filter instances for this graph type
        filtered_instances = filter(predicate, dataset.instances_map[graph_type])
        
        if !isempty(filtered_instances)
            filtered.instances_map[graph_type] = filtered_instances
        end
    end
    
    filtered.loaded = true
    return filtered
end

function get_instance(dataset::PidnebesnaDataset, instance_id::String)
    # Ensure dataset is loaded
    if !dataset.loaded
        load!(dataset)
    end
    
    # Search for instance in all graph types
    for instances in values(dataset.instances_map)
        for instance in instances
            if instance.id == instance_id
                return instance
            end
        end
    end
    
    throw(Common.ApproximateSymmetryError("Instance not found: $instance_id"))
end

function get_matrix_instance(dataset::PidnebesnaDataset, instance_id::String, simulation_index::Int=1)
    multi_instance = get_instance(dataset, instance_id)
    return get_simulation(multi_instance, simulation_index)
end

export PidnebesnaDataset, load!, get_graph_types, get_instance, get_matrix_instance