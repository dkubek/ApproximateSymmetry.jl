"""
    load_instance(path::String; sim_idx=nothing)

Load an instance from an NPZ file.
If sim_idx is provided, load just that simulation, otherwise load all simulations.
"""
function load_instance(path::String; sim_idx=nothing)
        if !isfile(path)
                error("File not found: $path")
        end

        # Extract base name from the path
        base_name = replace(basename(path), r"\.npz$" => "")

        # Load the NPZ file
        data = npzread(path)

        if sim_idx !== nothing
                # Load a specific simulation
                sim_key = string(sim_idx)
                if !haskey(data, sim_key)
                        error("Simulation $sim_idx not found in $path")
                end

                matrix = data[sim_key]
                id = "$(base_name)_sim$(sim_idx)"

                return MatrixInstance(id, matrix, Dict(:base_name => base_name, :simulation => sim_idx))
        else
                # Load all simulations
                instances = Vector{MatrixInstance}()

                for (key, matrix) in data
                        # Try to parse the key as an integer
                        try
                                sim_idx = parse(Int, key)
                                id = "$(base_name)_sim$(sim_idx)"
                                instance = MatrixInstance(id, matrix, Dict(:base_name => base_name, :simulation => sim_idx))
                                push!(instances, instance)
                        catch
                                # Skip keys that aren't valid simulation indices
                                continue
                        end
                end

                return instances
        end
end
