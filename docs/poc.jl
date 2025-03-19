begin
        using NPZ

        # Define the Instance struct with node count
        struct Instance
                graph_type::String
                path::String
                n::Int
        end

        function collect_instances(instance_folder::String)
                instances = Instance[]

                # Check if folder exists
                if !isdir(instance_folder)
                        error("Instance folder '$instance_folder' does not exist.")
                end

                # Iterate through subdirectories
                for graph_type_folder in readdir(instance_folder, join=true)
                        if isdir(graph_type_folder)
                                graph_type = basename(graph_type_folder)

                                # Iterate through files in the subdirectory
                                for instance_path in readdir(graph_type_folder, join=true)
                                        if isfile(instance_path) && endswith(instance_path, ".npz") && !contains(instance_path, "_allInfo")
                                                # Extract n_nodes from filename
                                                filename = basename(instance_path)
                                                n_nodes_match = match(r"nNodes(\d+)", filename)
                                                n_nodes = n_nodes_match !== nothing ? parse(Int, n_nodes_match.captures[1]) : 0

                                                push!(instances, Instance(graph_type, instance_path, n_nodes))
                                        end
                                end
                        end
                end

                return instances
        end

        """
            load_instance_data(instance::Instance) -> Dict

        Loads data from a .npz file and returns it as a dictionary.
        """
        function load_instance_data(instance::Instance)
                if !isfile(instance.path)
                        error("Instance file '$(instance.path)' does not exist.")
                end

                try
                        data = npzread(instance.path)
                        return data
                catch e
                        error("Failed to load instance data from '$(instance.path)': $e")
                end
        end
end

function S(A::AbstractMatrix, P::AbstractMatrix)
        n = size(A, 1)
        return norm(A - P * A * P', 2) / (n * (n - 1))
end

begin
        """
            get_perm_file_path(base_name, sim_idx, permutations_dir)

        Get the path to a permutation file.
        """
        function get_perm_file_path(base_name, sim_idx, permutations_dir)
                return joinpath(permutations_dir, "$(base_name)_sim$(sim_idx-1).csv")
        end

        """
            get_timing_file_path(base_name, sim_idx, timing_dir)

        Get the path to a timing file for a specific simulation.
        """
        function get_timing_file_path(base_name, sim_idx, timing_dir)
                return joinpath(timing_dir, "$(base_name)_sim$(sim_idx-1)_timing.csv")
        end

        """
            get_s_metric_file_path(base_name, sim_idx, s_dir)

        Get the path to an S metric file for a specific simulation.
        """
        function get_s_metric_file_path(base_name, sim_idx, s_dir)
                return joinpath(s_dir, "$(base_name)_sim$(sim_idx-1)_s_metric.csv")
        end

        """
            results_exist(base_name, sim_idx, permutations_dir, timing_dir, s_dir)

        Check if all result files for a simulation already exist.
        """
        function results_exist(base_name, sim_idx, permutations_dir, timing_dir, s_dir)
                perm_file = get_perm_file_path(base_name, sim_idx, permutations_dir)
                timing_file = get_timing_file_path(base_name, sim_idx, timing_dir)
                s_file = get_s_metric_file_path(base_name, sim_idx, s_dir)

                return isfile(perm_file) && isfile(timing_file) && isfile(s_file)
        end

        """
            save_timing_for_simulation(base_name, sim_idx, sim_times, timing_dir, num_runs)

        Save timing data for a single simulation.
        """
        function save_timing_for_simulation(base_name, sim_idx, sim_times, timing_dir, num_runs)
                timing_file = get_timing_file_path(base_name, sim_idx, timing_dir)

                # Create DataFrame for timing data
                time_df = DataFrame()
                time_df.run = 1:num_runs
                time_df.time = sim_times

                # Ensure the directory exists
                mkpath(dirname(timing_file))

                # Save to CSV
                CSV.write(timing_file, time_df)
        end

        """
            save_s_metric_for_simulation(base_name, sim_idx, s_values, s_dir, num_runs)

        Save S metric values for a single simulation.
        """
        function save_s_metric_for_simulation(base_name, sim_idx, s_values, s_dir, num_runs)
                s_file = get_s_metric_file_path(base_name, sim_idx, s_dir)

                # Create DataFrame for S metric data
                s_df = DataFrame()
                s_df.run = 1:num_runs
                s_df.s_metric = s_values

                # Ensure the directory exists
                mkpath(dirname(s_file))

                # Save to CSV
                CSV.write(s_file, s_df)
        end

        """
            process_instance_simulation(matrix, sim_idx, base_name, permutations_dir, timing_dir, s_dir, num_runs, method)

        Process a single simulation of an instance and save results.
        Uses the provided method to compute permutation matrices.
        """
        function process_instance_simulation(matrix, sim_idx, base_name, permutations_dir, timing_dir, s_dir, num_runs, method)
                # Run the algorithm multiple times
                sim_times = zeros(num_runs)
                sim_permutations = []
                sim_s_values = zeros(num_runs)

                for run in 1:num_runs
                        P, run_time = method(matrix)
                        sim_times[run] = run_time
                        push!(sim_permutations, P)

                        # Calculate S(A) metric
                        sim_s_values[run] = S(matrix, P)
                end

                # Save permutation matrices for this simulation
                perm_file = get_perm_file_path(base_name, sim_idx, permutations_dir)

                # Convert permutations to a format suitable for CSV
                n = size(sim_permutations[1], 1)
                perm_df = DataFrame()

                for run in 1:num_runs
                        P = sim_permutations[run]
                        # Find where each row has a 1 (the destination node)
                        destinations = [findfirst(x -> x == 1, P[i, :]) for i in 1:n]
                        perm_df[!, "run$run"] = destinations
                end

                # Ensure directories exist
                mkpath(dirname(perm_file))

                # Save permutation data
                CSV.write(perm_file, perm_df)

                # Save timing data
                save_timing_for_simulation(base_name, sim_idx, sim_times, timing_dir, num_runs)

                # Save S metric data
                save_s_metric_for_simulation(base_name, sim_idx, sim_s_values, s_dir, num_runs)

                return sim_times, sim_s_values
        end

        """
            load_existing_timing_data(timing_dir)

        Load existing timing data from individual timing files.
        """
        function load_existing_timing_data(timing_dir)
                timing_data = Dict()

                if !isdir(timing_dir)
                        return timing_data
                end

                for file in readdir(timing_dir)
                        if endswith(file, "_timing.csv")
                                # Extract base_name and sim_idx from filename
                                match_result = match(r"(.+)_sim(\d+)_timing\.csv", file)
                                if match_result !== nothing
                                        base_name = match_result.captures[1]
                                        sim_idx = parse(Int, match_result.captures[2]) + 1  # Convert from 0-based to 1-based

                                        # Initialize base_name dict if not exists
                                        if !haskey(timing_data, base_name)
                                                timing_data[base_name] = Dict()
                                        end

                                        # Read the timing file
                                        timing_file = joinpath(timing_dir, file)
                                        df = CSV.read(timing_file, DataFrame)

                                        # Store the timing data
                                        timing_data[base_name][sim_idx] = df.time
                                end
                        end
                end

                return timing_data
        end

        """
            load_existing_s_metric_data(s_dir)

        Load existing S metric data from individual files.
        """
        function load_existing_s_metric_data(s_dir)
                s_data = Dict()

                if !isdir(s_dir)
                        return s_data
                end

                for file in readdir(s_dir)
                        if endswith(file, "_s_metric.csv")
                                # Extract base_name and sim_idx from filename
                                match_result = match(r"(.+)_sim(\d+)_s_metric\.csv", file)
                                if match_result !== nothing
                                        base_name = match_result.captures[1]
                                        sim_idx = parse(Int, match_result.captures[2]) + 1  # Convert from 0-based to 1-based

                                        # Initialize base_name dict if not exists
                                        if !haskey(s_data, base_name)
                                                s_data[base_name] = Dict()
                                        end

                                        # Read the S metric file
                                        s_file = joinpath(s_dir, file)
                                        df = CSV.read(s_file, DataFrame)

                                        # Store the S metric data
                                        s_data[base_name][sim_idx] = df.s_metric
                                end
                        end
                end

                return s_data
        end

        """
            compile_and_save_timing_summary(timing_dir, output_file, num_runs)

        Compile all individual timing files into a summary CSV.
        """
        function compile_and_save_timing_summary(timing_dir, output_file, num_runs)
                # Load all timing data
                timing_data = load_existing_timing_data(timing_dir)

                # Pre-define column names
                col_names = ["base_name", "simulation"]
                append!(col_names, ["run$i" for i in 1:num_runs])

                # Initialize DataFrame with column names
                time_df = DataFrame([name => [] for name in col_names])

                # Collect all base names
                all_base_names = sort(collect(keys(timing_data)))

                # For each base name
                for base_name in all_base_names
                        # For each simulation
                        for sim_idx in sort(collect(keys(timing_data[base_name])))
                                # Create a row with base_name, simulation number, and times
                                row = Dict{String,Any}()
                                row["base_name"] = base_name
                                row["simulation"] = sim_idx - 1  # Convert to 0-based for output

                                for run in 1:num_runs
                                        if run <= length(timing_data[base_name][sim_idx])
                                                row["run$run"] = timing_data[base_name][sim_idx][run]
                                        else
                                                row["run$run"] = NaN  # Handle missing data
                                        end
                                end

                                push!(time_df, row)
                        end
                end

                # Save the summary CSV
                CSV.write(output_file, time_df)
        end

        """
            compile_and_save_s_metric_summary(s_dir, output_file, num_runs)

        Compile all individual S metric files into a summary CSV.
        """
        function compile_and_save_s_metric_summary(s_dir, output_file, num_runs)
                # Load all S metric data
                s_data = load_existing_s_metric_data(s_dir)

                # Pre-define column names
                col_names = ["base_name", "simulation"]
                append!(col_names, ["run$i" for i in 1:num_runs])

                # Initialize DataFrame with column names
                s_df = DataFrame([name => [] for name in col_names])

                # Collect all base names
                all_base_names = sort(collect(keys(s_data)))

                # For each base name
                for base_name in all_base_names
                        # For each simulation
                        for sim_idx in sort(collect(keys(s_data[base_name])))
                                # Create a row with base_name, simulation number, and S values
                                row = Dict{String,Any}()
                                row["base_name"] = base_name
                                row["simulation"] = sim_idx - 1  # Convert to 0-based for output

                                for run in 1:num_runs
                                        if run <= length(s_data[base_name][sim_idx])
                                                row["run$run"] = s_data[base_name][sim_idx][run]
                                        else
                                                row["run$run"] = NaN  # Handle missing data
                                        end
                                end

                                push!(s_df, row)
                        end
                end

                # Save the summary CSV
                CSV.write(output_file, s_df)
        end

        """
            process_dataset(input_dir, output_dir, method; method_name="ApproxSymmetry", method_version="v1", 
                            num_runs=5, force_recompute=false)

        Process all matrices in the dataset and save results in the specified format.
        Can resume from interrupted execution unless force_recompute is true.
        Uses the provided method function to compute permutation matrices.
        """
        function process_dataset(input_dir, output_dir, method;
                method_name="ApproxSymmetry",
                method_version="v1",
                num_runs=5,
                force_recompute=false)
                # Create output directories
                method_dir = joinpath(output_dir, method_name)
                version_dir = joinpath(method_dir, method_version)
                mkpath(version_dir)

                @info "Collecting instances from $input_dir"
                # Collect all instance data
                instances = collect_instances(input_dir)

                # Group instances by graph type
                graph_types = unique([inst.graph_type for inst in instances])

                for graph_type in graph_types
                        @info "Processing graph type: $graph_type"

                        # Get instances for this graph type
                        type_instances = filter(i -> i.graph_type == graph_type, instances)

                        # Create output directories for this model
                        model_output_dir = joinpath(version_dir, graph_type)
                        permutations_dir = joinpath(model_output_dir, "Permutations")
                        timing_dir = joinpath(model_output_dir, "Timing")
                        s_dir = joinpath(model_output_dir, "S_Metric")
                        mkpath(permutations_dir)
                        mkpath(timing_dir)
                        mkpath(s_dir)

                        for instance in type_instances
                                # Extract base filename without extension
                                base_name = replace(basename(instance.path), r"\.npz$" => "")

                                @info "  Processing instance: $base_name"

                                # Load the data only if we need to process at least one simulation
                                data = nothing

                                # Process each simulation
                                for sim_idx in 1:39
                                        # Check if results already exist
                                        if !force_recompute && results_exist(base_name, sim_idx, permutations_dir, timing_dir, s_dir)
                                                @info "    Simulation $(sim_idx-1): Already processed, skipping"
                                                continue
                                        end

                                        # Lazy loading of data
                                        if data === nothing
                                                @debug "    Loading data for instance $base_name"
                                                data = load_instance_data(instance)
                                        end

                                        @info "    Simulation $(sim_idx-1): Processing..."

                                        matrix = data["$(sim_idx-1)"]

                                        # Process the simulation and save results
                                        process_instance_simulation(
                                                matrix, sim_idx, base_name, permutations_dir, timing_dir, s_dir, num_runs, method
                                        )
                                end
                        end

                        # Compile and save timing summary for this graph type
                        @info "Compiling timing summary for $graph_type"
                        time_summary_file = joinpath(model_output_dir, "$(method_name)_time_$(graph_type).csv")
                        compile_and_save_timing_summary(timing_dir, time_summary_file, num_runs)

                        # Compile and save S metric summary for this graph type
                        @info "Compiling S metric summary for $graph_type"
                        s_summary_file = joinpath(model_output_dir, "$(method_name)_s_metric_$(graph_type).csv")
                        compile_and_save_s_metric_summary(s_dir, s_summary_file, num_runs)
                end

                @info "Processing completed."
        end
end
