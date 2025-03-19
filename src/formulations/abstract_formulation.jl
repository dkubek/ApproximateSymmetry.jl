"""
    AbstractFormulation

Abstract type representing a formulation of the approximate symmetry problem.
All concrete formulation implementations should be subtypes of this.
"""
abstract type AbstractFormulation end

"""
    get_objective_function(formulation::AbstractFormulation)

Get the objective function associated with this formulation.
"""
function get_objective_function(formulation::AbstractFormulation)
    error("get_objective_function not implemented for $(typeof(formulation))")
end

"""
    solve(formulation::AbstractFormulation, instance)

Solve the approximate symmetry problem using the given formulation and instance.
This is the main entry point for solving problems.

Returns a NamedTuple with the solution details.
"""
function solve(formulation::AbstractFormulation, instance)
    error("solve method not implemented for $(typeof(formulation))")
end

"""
    objective_value(formulation::AbstractFormulation, x, instance)

Compute the objective function value for the given formulation,
variable values x, and problem instance.
"""
function objective_value(formulation::AbstractFormulation, x, instance)
    obj_func = get_objective_function(formulation)
    return evaluate(obj_func, x, instance)
end

export AbstractFormulation, get_objective_function, solve, objective_value
