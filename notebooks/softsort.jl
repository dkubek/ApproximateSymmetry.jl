### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 28275e76-80c9-4c32-be2a-1d747bd4e575
begin
    import Pkg
    package_path = joinpath(@__DIR__, "..")
    
	# Create a temporary environment
    Pkg.activate(mktempdir())
    
	# Add local package in development mode
    Pkg.develop(path=package_path)

	using ApproximateSymmetry
end

# ╔═╡ e0ebc5c8-bc49-4a04-af2f-3809b27cd03e
begin
	# Add other dependencies
    Pkg.add([
		"DataFrames",

		"LinearAlgebra",
		"SparseArrays",
		"LazyArrays",

		"Optimization",
		"OptimizationOptimJL",
		"Hungarian",

		"NNlib",

		"Distributions",
		"HyperTuning",
		
		"Plots",
	])
end

# ╔═╡ 22e7331d-ea12-4d7e-9b06-a74c362a4dff
begin
	using LinearAlgebra
	using SparseArrays
	using LazyArrays

	using Plots
	#theme(:ggplot2)  # Use the ggplot2 theme

	using Optimization, OptimizationOptimJL
	using Hungarian

	using Random
end

# ╔═╡ e037c2aa-2b3f-43c5-af35-85012fffc322
begin
	using NNlib
	function softsort(x::AbstractVector{T}; τ=0.1) where T
	    n = length(x)

	    diff_matrix = -abs.(sort(x) * ones(n)' - ones(n) * x')
	    scaled_diffs = diff_matrix / τ

		# Protect against overflow with clipping
    	max_val = 700.0  # near log(Float64_MAX)
    	scaled_diffs = clamp.(scaled_diffs, -max_val, max_val)

	    softmax(scaled_diffs)
	end
end

# ╔═╡ 6053698c-e97d-45c1-ae25-fb83db26d009
n = 4

# ╔═╡ cf2e32ba-4298-47c9-b053-f25106466b34
x = rand(n)

# ╔═╡ 8b451c89-61d9-485b-be23-b2ccc81dec25
softsort(x)

# ╔═╡ 761b869b-59fd-44dd-aec9-962defecb307
begin
	function objective(x, p)
	    A, c, τ = p   
	    P = softsort(x; τ=τ)    
	    return -tr(A * P * A * P') + tr(Diagonal(c) * P)
	end
	
	function optimize(A, c; x0=nothing, τ=0.01, maxiters=1000)
	    n = size(A, 1)
		
		if isnothing(x0)
	        x0 = randn(n)
	    end
	    
	    p = (A, c, τ)
	    
	    optf = OptimizationFunction(objective, Optimization.AutoForwardDiff())
	    prob = OptimizationProblem(optf, x0, p)
		
	    sol = SciMLBase.solve(prob, BFGS(), maxiters=maxiters)
	    
	    return sol
	end
end

# ╔═╡ b60e8241-3109-4bfb-9a30-26868c1a164f
ds = PidnebesnaDataset("../data/pidnebesna") |> load!

# ╔═╡ be216121-5780-4e6b-9e1c-0037e9f57984
instance = Iterators.filter(ds) do instance
	contains(instance.id, "nNodes50")
end |> first

# ╔═╡ b6b59864-8e67-424f-a01b-1f57ead6ef00
A = adjacency(instance)

# ╔═╡ cdca8d34-9c11-4d7c-9b0c-d4449e241957
A |> heatmap

# ╔═╡ 06d4d369-6c94-47c0-bbe5-a8a609665115
1000:5:20000 |> collect

# ╔═╡ a85f9a74-6ecb-43ef-a2be-0bf2855a66e3
begin
	@unpack c, τ = scenario
end

# ╔═╡ 07763fd6-e698-4a79-8281-1f9b08fd0279
begin
	c_best = 0.00215443
	τ_best = 0.1
end

# ╔═╡ 82e8caca-a2ca-46e6-9c86-6e52752fad2d
B = Iterators.filter(ds) do instance
	contains(instance.id, "LRM") &&
	contains(instance.id, "nNodes100")
end |> first |> adjacency

# ╔═╡ c3a3a6bd-d2ac-46aa-880b-2e2260dc2b70
#=╠═╡
function annealing(A, c;
	x0=nothing,
	τ_init=0.1, τ_final=0.0001, annealing_steps=5,
	maxiters=1000
)
	n = size(A, 1)
	
	if isnothing(x0)
		x0 = randn(n)
	end

	# Define temperature schedule (exponential decay)
    τ_schedule = exp.(range(log(τ_init), log(τ_final), length=annealing_steps))

	for (step, τ) in enumerate(τ_schedule)
        println("Annealing step $step/$annealing_steps with τ = $τ")
        sol = optimize(A, c; x0=x0, τ=τ)
		x0 = sol.u
	end
	
	return sol
end
  ╠═╡ =#

# ╔═╡ 45d1147e-1f34-488a-9a19-7d54cc5937d0
#=╠═╡
sol.u |> softsort
  ╠═╡ =#

# ╔═╡ 3a9cb567-2f97-47a4-8123-3abe55488a1f
P_anneal = let
	SS = sol_anneal.u |> normalize |> softsort
	Hungarian.munkres(SS) .== 2
end

# ╔═╡ 7ee2b63d-de6a-4342-8acb-60905dcc09e0
#=╠═╡
sol.stats.time
  ╠═╡ =#

# ╔═╡ 58581ed0-eecb-4ac8-be1c-485e3e02697a
function soft_sort(s, tau)
    # Get dimensions
    batch_size, n, dim = size(s)
    
    # Sort s along dim=2 (descending)
    s_sorted = sort(s, dims=2, rev=true)
    
    # Initialize pairwise distances
    pairwise_distances = zeros(Float64, batch_size, n, n)
    
    # Compute pairwise distances
    @inbounds for b in 1:batch_size
        for i in 1:n, j in 1:n
            dist = 0.0
            for d in 1:dim
                dist -= abs(s[b, i, d] - s_sorted[b, j, d])
            end
            pairwise_distances[b, i, j] = dist
        end
    end
    
    # Apply softmax with temperature
    P_hat = similar(pairwise_distances)
    @inbounds for b in 1:batch_size
        for i in 1:n
            # Compute softmax for numerical stability
            scaled = view(pairwise_distances, b, i, :) ./ tau
            max_val = maximum(scaled)
            exp_vals = exp.(scaled .- max_val)
            sum_exp = sum(exp_vals)
            P_hat[b, i, :] = exp_vals ./ sum_exp
        end
    end
    
    return P_hat
end

# ╔═╡ 9502c291-9810-4fc7-ac16-e1113439ea67
#=╠═╡
soft_sort(sol.u, 0.1)
  ╠═╡ =#

# ╔═╡ 95f8d43c-f0db-40ac-be8b-a1b30e9191e2
md"""
---
"""

# ╔═╡ 24b76b60-565e-4cdf-9d0d-0c7d0c669baf
function method_softsort(A::AbstractMatrix)
	n = size(A, 1)
	sol = optimize(A, 0.2 * ones(n))

	P = Hungarian.munkres(- sol.u |> softsort) .== 2
	P, sol.stats.time
end

# ╔═╡ 2f9d696c-9d18-4518-b32b-15a8ec52306f
#=╠═╡
function method_anneal(A::AbstractMatrix)
	n = size(A, 1)
	sol = annealing(A, 0.2 * ones(n))

	P = Hungarian.munkres(- sol.u |> softsort) .== 2
	P, sol.stats.time
end
  ╠═╡ =#

# ╔═╡ e7e48338-9d05-481a-a86a-84979771f576
E(A, P) = 1/4 * norm(A - P * A * P', 1)

# ╔═╡ b13ef39d-4de3-4727-b1f6-758f2d37edd8
let
	nruns = 5
	n = size(A, 1)

	c = 0.1
	τ = 0.01
	maxiters = 10000

	Es = Vector{Float64}(undef, nruns)
	for i in eachindex(Es)
		sol = optimize(A, c * ones(n), τ=τ, maxiters=maxiters)
		P = Hungarian.munkres(- sol.u |> softsort) .== 2
		Es[i] = E(A, P)
	end
	minimum(Es)
end

# ╔═╡ 5a0942a6-add5-4450-89fb-08868a812ead
begin
	using HyperTuning
	import Base: LogRange
	
	function objective(trial)
	    @suggest c in trial
	    @suggest τ in trial
	    
		#@suggest maxiters in trial
		maxiters=10000
		
		#@suggest nruns in trial
		nruns = 5
	    
	    n = size(A, 1)
	    Es = Vector{Float64}(undef, nruns)

		i = 0
		while i < nruns
	        sol = optimize(A, c * ones(n), τ=τ, maxiters=maxiters)
	        P = Hungarian.munkres(- sol.u |> softsort) .== 2
	        
			if tr(P) != n
				i += 1
			else
				continue
			end

			Es[i] = E(A, P)
	    end
	    
	    # Return minimum energy (assuming lower is better)
	    return minimum(Es)
	end

end

# ╔═╡ 76d26377-0474-4b13-9702-4aa64f8dfb16
begin
	scenario = HyperTuning.Scenario(
		c = LogRange(0.001, 1.0, 10),
		τ = LogRange(0.00001, 0.1, 10),
		#maxiters = [1000, 5000],
	)
	HyperTuning.optimize(objective, scenario)
end

# ╔═╡ ce9d492b-30a5-49e1-8f3a-dd400529abc6
begin
	@info "Top parameters"
	top_parameters(scenario)
end

# ╔═╡ b9d2c2f3-c65a-40af-b47b-e5c62941f104
begin
	@info "History"
	display(history(scenario))
end

# ╔═╡ 66aa1350-cc08-4c37-a2e4-bba772493eb4
#=╠═╡
E(B, P)
  ╠═╡ =#

# ╔═╡ 1871705c-a4f3-4510-b1b0-da9056d0ee05
#=╠═╡
sol = let
	n = size(B, 1)
	optimize(B, 0.2 * ones(n), τ=τ_best)
end
  ╠═╡ =#

# ╔═╡ a7554d3d-d274-4339-9826-6f8d5db33d1f
# ╠═╡ disabled = true
#=╠═╡
P = Hungarian.munkres(- sol.u |> softsort) .== 2
  ╠═╡ =#

# ╔═╡ 0d8ed925-4fc0-4c95-9b7d-a65e12eaa90b
# ╠═╡ disabled = true
#=╠═╡
sol = let
	n = size(A, 1)
	optimize(A, 0.2 * ones(n))
end
  ╠═╡ =#

# ╔═╡ d83686a4-aa08-4fe0-a2a0-ef893d49d078
#=╠═╡
P = Hungarian.munkres(- sol.u |> softsort) .== 2
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═28275e76-80c9-4c32-be2a-1d747bd4e575
# ╠═e0ebc5c8-bc49-4a04-af2f-3809b27cd03e
# ╠═22e7331d-ea12-4d7e-9b06-a74c362a4dff
# ╠═e037c2aa-2b3f-43c5-af35-85012fffc322
# ╠═6053698c-e97d-45c1-ae25-fb83db26d009
# ╠═cf2e32ba-4298-47c9-b053-f25106466b34
# ╠═8b451c89-61d9-485b-be23-b2ccc81dec25
# ╠═761b869b-59fd-44dd-aec9-962defecb307
# ╠═c3a3a6bd-d2ac-46aa-880b-2e2260dc2b70
# ╠═b60e8241-3109-4bfb-9a30-26868c1a164f
# ╠═be216121-5780-4e6b-9e1c-0037e9f57984
# ╠═b6b59864-8e67-424f-a01b-1f57ead6ef00
# ╠═cdca8d34-9c11-4d7c-9b0c-d4449e241957
# ╠═0d8ed925-4fc0-4c95-9b7d-a65e12eaa90b
# ╠═45d1147e-1f34-488a-9a19-7d54cc5937d0
# ╠═a7554d3d-d274-4339-9826-6f8d5db33d1f
# ╠═b13ef39d-4de3-4727-b1f6-758f2d37edd8
# ╠═06d4d369-6c94-47c0-bbe5-a8a609665115
# ╠═5a0942a6-add5-4450-89fb-08868a812ead
# ╠═76d26377-0474-4b13-9702-4aa64f8dfb16
# ╠═ce9d492b-30a5-49e1-8f3a-dd400529abc6
# ╠═b9d2c2f3-c65a-40af-b47b-e5c62941f104
# ╠═a85f9a74-6ecb-43ef-a2be-0bf2855a66e3
# ╠═07763fd6-e698-4a79-8281-1f9b08fd0279
# ╠═82e8caca-a2ca-46e6-9c86-6e52752fad2d
# ╠═1871705c-a4f3-4510-b1b0-da9056d0ee05
# ╠═d83686a4-aa08-4fe0-a2a0-ef893d49d078
# ╠═66aa1350-cc08-4c37-a2e4-bba772493eb4
# ╠═3a9cb567-2f97-47a4-8123-3abe55488a1f
# ╠═7ee2b63d-de6a-4342-8acb-60905dcc09e0
# ╠═58581ed0-eecb-4ac8-be1c-485e3e02697a
# ╠═9502c291-9810-4fc7-ac16-e1113439ea67
# ╠═95f8d43c-f0db-40ac-be8b-a1b30e9191e2
# ╠═24b76b60-565e-4cdf-9d0d-0c7d0c669baf
# ╠═2f9d696c-9d18-4518-b32b-15a8ec52306f
# ╠═e7e48338-9d05-481a-a86a-84979771f576
