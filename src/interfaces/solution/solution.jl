"""
    Solution{T} <: AbstractSolution

Concrete implementation of AbstractSolution.

# Fields
- `result::T`: The actual solution/output
- `metrics::Dict{Symbol,Any}`: Dictionary of metrics computed during solving
"""
struct Solution{T} <: AbstractSolution
    result::T
    metrics::Dict{Symbol,Any}
end

"""
    Solution(result::T) where {T}

Construct a solution with the given result and empty metrics.
"""
Solution(result::T) where {T} = Solution{T}(result, Dict{Symbol,Any}())

"""
    get_result(solution::Solution)

Get the actual result from the solution.
"""
get_result(solution::Solution) = solution.result

"""
    get_metric(solution::Solution, key::Symbol, default=nothing)

Get a metric value from the solution with an optional default value.
"""
get_metric(solution::Solution, key::Symbol, default=nothing) = get(solution.metrics, key, default)

"""
    has_metric(solution::Solution, key::Symbol)

Check if a metric exists in the solution.
"""
has_metric(solution::Solution, key::Symbol) = haskey(solution.metrics, key)

"""
    set_metric!(solution::Solution, key::Symbol, value)

Set a metric value in the solution.
"""
function set_metric!(solution::Solution, key::Symbol, value)
    solution.metrics[key] = value
    return solution
end

"""
    get_metrics(solution::Solution)

Get all metrics from the solution.
"""
get_metrics(solution::Solution) = solution.metrics

# Export the solution types and functions
export Solution, get_result, get_metric, has_metric, set_metric!, get_metrics

