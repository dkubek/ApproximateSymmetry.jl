using NPZ

mutable struct PidnebesnaSimulation{T<:Real} <: AbstractInstance
    id::String
    path::String
    simulation_index::Int

    loaded::Bool
    matrix::Union{Nothing,Matrix{T}}

    metadata::Dict{Symbol,Any}

    function PidnebesnaSimulation{T}(id::String, path::String, simulation_index::Int) where {T<:Real}
        new{T}(id, path, simulation_index, false, nothing, Dict{Symbol,Any}())
    end
end

function PidnebesnaSimulation(id::String, path::String, simulation_index::Int, ::Type{T}=Float64) where {T<:Real}
    PidnebesnaSimulation{T}(id, path, simulation_index)
end

function PidnebesnaSimulation(id::String, path::String, simulation_index::Int, matrix::Matrix{T}) where {T<:Real}
    sim = PidnebesnaSimulation{T}(id, path, simulation_index)
    sim.matrix = matrix
    sim.loaded = true
    return sim
end

function Instances.adjacency(instance::PidnebesnaSimulation{T}) where {T<:Real}
    if !instance.loaded
        load!(instance)
    end

    return instance.matrix
end

function load!(instance::PidnebesnaSimulation{T}) where {T<:Real}
    if instance.loaded
        return instance
    end

    data = NPZ.npzread(instance.path)

    # Extract number of nodes if available in filename
    # FIXME: Extract other statistics as well
    # n_nodes = 0
    # n_nodes_match = match(r"nNodes(\d+)", base_name)
    # if n_nodes_match !== nothing
    #         n_nodes = parse(Int, n_nodes_match.captures[1])
    # end

    sim_key = string(instance.simulation_index)

    # Check if this simulation exists
    if !haskey(data, sim_key)
        @error "The NPZ file $(instance.path) does not contain simulation index $(instance.simulation_index)"
    end

    # Get matrix for this simulation
    matrix = convert(Matrix{T}, data[sim_key])

    # Verify matrix dimensions
    #if n_nodes > 0 && size(matrix, 1) != n_nodes
    #        @warn "Matrix dimensions ($(size(matrix, 1))) do not match expected size ($n_nodes) for simulation $sim_key in $base_name"
    #end

    instance.matrix = matrix
    instance.loaded = true

    return instance
end

struct PidnebensInstance
    id::String
    graph_type::String
    path::String

    simulations::Vector{PidnebesnaSimulation}

    function PidnebensInstance(path::String, graph_type::String)
        base_name = basename(path)
        instance_id = replace(base_name, r"\.npz$" => "")

        simulations = Vector{PidnebesnaSimulation}()
        for sim_idx in 0:38
            sim_id = "$(instance_id)_sim$(sim_idx)"

            # Create simulation instance
            simulation = PidnebesnaSimulation(sim_id, path, sim_idx)

            simulation.metadata[:graph_type] = graph_type
            simulation.metadata[:base_instance] = instance_id

            push!(simulations, simulation)
        end

        new(instance_id, graph_type, path, simulations)
    end
end

Base.length(instance::PidnebensInstance) = length(instance.simulations)

Base.iterate(instance::PidnebensInstance) = iterate(instance.simulations)
Base.iterate(instance::PidnebensInstance, state) = iterate(instance.simulations, state)
Base.eltype(::Type{PidnebensInstance}) = PidnebesnaSimulation


mutable struct PidnebesnaDataset <: AbstractDataset
    path::String

    graph_types::Vector{String}
    instances::Vector{PidnebensInstance}
    loaded::Bool

    metadata::Dict{Symbol,Any}

    function PidnebesnaDataset(path::String)
        if !isdir(path)
            throw(Common.ApproximateSymmetryError("Dataset directory does not exist: $path"))
        end

        new(path, String[], PidnebensInstance[], false, Dict{Symbol,Any}())
    end
end

function load!(dataset::PidnebesnaDataset; force=false)
    if force
        empty!(dataset.graph_types)
        empty!(dataset.instances)
        empty!(dataset.metadata)
        dataset.loaded = false
    end

    if dataset.loaded
        return dataset
    end

    graph_type_dirs = filter(isdir, readdir(dataset.path, join=true))

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

    if !(graph_type in dataset.graph_types)
        push!(dataset.graph_types, graph_type)
    end

    # Find all NPZ files in the directory
    npz_files = filter(readdir(graph_type_dir, join=true)) do f
        isfile(f) && endswith(f, ".npz") && !contains(f, "_allInfo")
    end

    for npz_file in npz_files
        instance = PidnebensInstance(npz_file, graph_type)
        push!(dataset.instances, instance)
    end

    return dataset
end


function Base.length(dataset::PidnebesnaDataset)
    load!(dataset)

    map(dataset.instances) do instance
        length(instance)
    end |> sum
end

function Base.iterate(dataset::PidnebesnaDataset)
    load!(dataset)

    flat_iter = Iterators.flatten(dataset.instances)
    result = iterate(flat_iter)

    if result === nothing
        return nothing
    end

    simulation, state = result
    return simulation, (flat_iter, state)
end

function Base.iterate(::PidnebesnaDataset, state)
    flat_iter, iter_state = state
    result = iterate(flat_iter, iter_state)

    if result === nothing
        return nothing
    end

    simulation, new_state = result
    return simulation, (flat_iter, new_state)
end
Base.eltype(::Type{PidnebesnaDataset}) = PidnebesnaSimulation


export PidnebesnaSimulation, adjacency, load!, PidnebesnaDataset, load!
