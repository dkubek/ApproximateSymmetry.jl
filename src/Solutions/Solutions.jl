module Solutions

using LinearAlgebra
using ..Common
using ..Instances

struct Solution{T <: Real}
	permutation::Matrix{T}
	metrics::Dict{Symbol, Any}
	instance_id::String

	function Solution(permutation::Matrix{T}, metrics::Dict{Symbol, Any}, instance_id::String) where {T <: Real}
		n, m = size(permutation)
		if n != m
			throw(Common.ApproximateSymmetryError("Permutation matrix must be square: got dimensions $n x $m"))
		end

		# Ensure required metrics are present
		if !haskey(metrics, :time)
			metrics[:time] = NaN
		end

		return new{T}(permutation, metrics, instance_id)
	end
end

function Solution(permutation::Matrix{T}, instance::I) where {T <: Real, I <: AbstractInstance}
	metrics = Dict{Symbol, Any}(:time => NaN)
	return Solution(permutation, metrics, instance.id)
end

function get_result(solution::Solution)
	return solution.permutation
end

function get_metric(solution::Solution, key::Symbol, default = nothing)
	return get(solution.metrics, key, default)
end

function has_metric(solution::Solution, key::Symbol)
	return haskey(solution.metrics, key)
end

function set_metric!(solution::Solution, key::Symbol, value)
	solution.metrics[key] = value
	return solution
end

function get_metrics(solution::Solution)
	return solution.metrics
end

export Solution
export get_result, get_metric, has_metric, set_metric!, get_metrics

end # module