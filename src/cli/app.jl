module CLI

using ArgParse
using ..ApproximateSymmetry

"""
    parse_commandline()

Parse command line arguments for the approximate symmetry application.
"""
function parse_commandline()
    s = ArgParseSettings(
        description="ApproximateSymmetry - A package for approximate symmetry analysis",
        version="0.1.0",
        add_version=true
    )

    @add_arg_table s begin
        "command"
        help = "Command to execute (process, solve)"
        arg_type = String
        required = true
        "--dataset", "-d"
        help = "Dataset type to use (e.g., NPZDataset)"
        arg_type = String
        default = "NPZDataset"
        "--path", "-p"
        help = "Path to the dataset or instance"
        arg_type = String
        required = true
        "--method", "-m"
        help = "Method to use for solving (e.g., SimpleMethod)"
        arg_type = String
        default = "SimpleMethod"
        "--output", "-o"
        help = "Output directory"
        arg_type = String
        default = "results"
        "--format", "-f"
        help = "Output format (e.g., CSVOutputFormat)"
        arg_type = String
        default = "CSVOutputFormat"
        "--runs", "-r"
        help = "Number of runs per instance"
        arg_type = Int
        default = 5
        "--force"
        help = "Force recomputation of existing results"
        action = :store_true
        "--instance", "-i"
        help = "Specific instance to process (optional)"
        arg_type = String
        "--parameters", "-P"
        help = "Method parameters as key=value pairs (e.g., -P weight=0.5 -P tolerance=1e-6)"
        arg_type = String
        nargs = '+'
        default = String[]
        "--graph-type", "-g"
        help = "Graph type for NPZ datasets"
        arg_type = String
    end

    return parse_args(s)
end

"""
    get_dataset(dataset_type::String, path::String, graph_type::Union{String,Nothing})

Get a dataset of the specified type.
"""
function get_dataset(dataset_type::String, path::String, graph_type::Union{String,Nothing})
    # Get the dataset type as a Julia type
    dataset_type_sym = Symbol(dataset_type)
    if !isdefined(ApproximateSymmetry, dataset_type_sym)
        error("Unknown dataset type: $dataset_type")
    end

    dataset_type_julia = getfield(ApproximateSymmetry, dataset_type_sym)

    # Read the dataset
    if dataset_type == "NPZDataset" && graph_type !== nothing
        # For NPZ datasets, we need a graph type
        dataset = NPZDataset(path, graph_type)
        return dataset
    else
        # For other dataset types, use the generic read_dataset function
        datasets = read_dataset(dataset_type_julia, path)
        if isa(datasets, Dict) && !isempty(datasets)
            # If we got a dictionary of datasets, return the first one
            return first(values(datasets))
        else
            return datasets
        end
    end
end

"""
    get_method(method_type::String, parameters::Vector{String})

Get a method of the specified type with the given parameters.
"""
function get_method(method_type::String, parameters::Vector{String})
    # Get the method type as a Julia type
    method_type_sym = Symbol(method_type)
    if !isdefined(ApproximateSymmetry, method_type_sym)
        error("Unknown method type: $method_type")
    end

    # Create a method of the specified type
    if method_type == "SimpleMethod"
        # For SimpleMethod, we need a solver function
        method = SimpleMethod(method_type, "v1", identity)
    else
        # For other method types, assume a constructor without arguments
        method_type_julia = getfield(ApproximateSymmetry, method_type_sym)
        method = method_type_julia()
    end

    # Set parameters
    for param in parameters
        key_value = split(param, "=")
        if length(key_value) != 2
            error("Invalid parameter format: $param. Expected key=value")
        end

        key = Symbol(key_value[1])
        value_str = key_value[2]

        # Try to parse the value as a number
        value = tryparse(Float64, value_str)
        if value === nothing
            # If not a number, use the string value
            value = value_str
        end

        set_parameter!(method, key, value)
    end

    return method
end

"""
    get_output_format(format_type::String)

Get an output format of the specified type.
"""
function get_output_format(format_type::String)
    # Get the format type as a Julia type
    format_type_sym = Symbol(format_type)
    if !isdefined(ApproximateSymmetry, format_type_sym)
        error("Unknown output format type: $format_type")
    end

    format_type_julia = getfield(ApproximateSymmetry, format_type_sym)
    return format_type_julia()
end

"""
    process_command(args)

Process the command specified in the arguments.
"""
function process_command(args)
    command = args["command"]

    if command == "process"
        # Get the dataset
        dataset = get_dataset(args["dataset"], args["path"], args["graph-type"])

        # Get the method
        method = get_method(args["method"], args["parameters"])

        # Get the output format
        format = get_output_format(args["format"])

        # Process the dataset
        process_dataset(dataset, method, args["output"], format;
            num_runs=args["runs"], force_recompute=args["force"])

        println("Processing complete. Results saved to $(args["output"])")
    elseif command == "solve"
        if args["instance"] === nothing
            error("No instance specified for 'solve' command")
        end

        # Load the instance
        instance = load_instance(args["instance"])

        # Get the method
        method = get_method(args["method"], args["parameters"])

        # Get the output format
        format = get_output_format(args["format"])

        # Solve the instance
        solution = solve(method, instance)

        # Write the solution
        output_file = joinpath(args["output"], "solution.csv")
        write_solution(solution, format, output_file)

        println("Solution saved to $output_file")
    else
        error("Unknown command: $command")
    end
end

"""
    main()

Main entry point for the CLI application.
"""
function main()
    try
        args = parse_commandline()
        process_command(args)
    catch e
        println("Error: $(sprint(showerror, e))")
        exit(1)
    end
end

end # module CLI

export CLI
