"""
    solve(method::AbstractMethod, instance::AbstractInstance)

Apply a solution method to an instance and return the result.
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
"""
struct SimpleMethod <: AbstractMethod
    name::String
    version::String
    solver_func::Function
    parameters::Dict{Symbol,Any}
end

"""
    SimpleMethod(name::String, version::String, solver_func::Function)

Construct a simple method with the given name, version, and solver function.
"""
SimpleMethod(name::String, version::String, solver_func::Function) =
    SimpleMethod(name, version, solver_func, Dict{Symbol,Any}())

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
    solve(method::SimpleMethod, instance::MatrixInstance)

Apply a simple method to a matrix instance.
"""
function solve(method::SimpleMethod, instance::MatrixInstance)
    # Time the execution
    start_time = time()

    # Call the solver function
    result = method.solver_func(instance.matrix)

    # Calculate execution time
    execution_time = time() - start_time

    # Return the result and execution time
    return result, execution_time
end
