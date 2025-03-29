"""
    solve(method::AbstractMethod, instance::AbstractInstance)

Apply a solution method to an instance and return the result as a Solution object.
"""
function solve end

"""
    SimpleMethod <: AbstractMethod

A basic implementation of the AbstractMethod interface.

# Fields
- `name::String`: Name of the method
- `version::String`: Version identifier
- `solver_func::Function`: The actual solving function
- `parameters::Dict{Symbol, Any}`: Additional parameters
- `supported_metrics::Vector{Symbol}`: Metrics that this method can compute
"""
struct SimpleMethod <: AbstractMethod
    name::String
    version::String
    solver_func::Function
    parameters::Dict{Symbol,Any}
    supported_metrics::Vector{Symbol}
end

"""
    SimpleMethod(name::String, version::String, solver_func::Function)

Construct a simple method with the given name, version, and solver function.
By default, only the :time metric is supported.
"""
SimpleMethod(name::String, version::String, solver_func::Function) =
    SimpleMethod(name, version, solver_func, Dict{Symbol,Any}(), Symbol[:time])

"""
    set_parameter!(method::SimpleMethod, key::Symbol, value)

Set a parameter for the method.
"""
function set_parameter!(method::SimpleMethod, key::Symbol, value)
    method.parameters[key] = value
    return method
end

"""
    get_parameter(method::SimpleMethod, key::Symbol, default=nothing)

Get a parameter from the method with an optional default value.
"""
function get_parameter(method::SimpleMethod, key::Symbol, default=nothing)
    get(method.parameters, key, default)
end

"""
    add_supported_metric!(method::SimpleMethod, metric::Symbol)

Add a metric to the list of supported metrics for this method.
"""
function add_supported_metric!(method::SimpleMethod, metric::Symbol)
    if !(metric in method.supported_metrics)
        push!(method.supported_metrics, metric)
    end
    return method
end

"""
    supported_metrics(method::SimpleMethod)

Get all metrics supported by this method.
"""
function supported_metrics(method::SimpleMethod)
    return method.supported_metrics
end

"""
    solve(method::SimpleMethod, instance::MatrixInstance)

Apply a simple method to a matrix instance.
Returns a Solution object containing the result and computed metrics.
"""
function solve(method::SimpleMethod, instance::MatrixInstance)
    # Time the execution
    start_time = time()

    # Call the solver function
    result = method.solver_func(instance.matrix)

    # Calculate execution time
    execution_time = time() - start_time

    # Create solution object and add metrics
    solution = Solution(result)

    # Add time metric if supported
    if :time in supported_metrics(method)
        set_metric!(solution, :time, execution_time)
    end

    return solution
end

export solve