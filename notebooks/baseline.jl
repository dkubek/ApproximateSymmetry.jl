### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 95f44e1a-1644-11f0-3029-5f3b87f55957
begin
    import Pkg
    package_path = joinpath(@__DIR__, "..")
    
	# Create a temporary environment
    Pkg.activate(mktempdir())
    
	# Add local package in development mode
    Pkg.develop(path=package_path)

	using ApproximateSymmetry
end

# ╔═╡ 18c9368b-1964-452b-a9fb-4bd716c327f5
begin
	# Add other dependencies
    Pkg.add([
		"DataFrames",

		"LinearAlgebra",
		"SparseArrays",
		"LazyArrays",

		"Optimization",
		"OptimizationMOI",
		"Ipopt",
		"MKL",
		"Hungarian",

		"Zygote",
		
		"Plots",
		"StatsPlots",
	])
end

# ╔═╡ a62c3acd-c47c-48fa-99fd-da9000c4a232
begin
	using LinearAlgebra
	using SparseArrays
	using LazyArrays

	using Plots
	using StatsPlots
	#theme(:ggplot2)  # Use the ggplot2 theme
	using DataFrames
	using Statistics

	using Optimization, OptimizationMOI, MKL, Ipopt
	using Hungarian

	using Random
end

# ╔═╡ 8daedc07-df00-46ad-835b-d6fc70d89575
using Zygote

# ╔═╡ 16f8ace3-042c-4d45-9c2a-ad539bea6d13
ds = PidnebesnaDataset("../data/pidnebesna") |> load!

# ╔═╡ e7cc1995-38aa-4304-b8e1-46e31b86ffd5
instance = Iterators.filter(ds) do instance
	contains(instance.id, "nNodes50")
end |> first

# ╔═╡ 0e09b4c0-4006-43f5-9160-ef41310e11e1
A = adjacency(instance)

# ╔═╡ 9cb38998-a485-41f4-9dd5-35759875cf22
n = size(A, 1)

# ╔═╡ 5cfb8286-0975-40a8-96e1-5c5d7b42a7d5
reshape(A, n * n)

# ╔═╡ b6a1cc10-3b3f-4d23-85dc-4c31f0420e7c
begin

	"""
    unpack_parameters!(P::AbstractMatrix, x::AbstractVector, n::Int)
	
	Unpacks a flattened vector `x` representing a doubly stochastic matrix into matrix `P`.
	Efficiently copies data without unnecessary allocations.
	"""
	function unpack_parameters!(P::AbstractMatrix, x::AbstractVector, n::Int)
	    copyto!(P, reshape(x, n, n))
	    return P
	end

	function unpack_parameters(x::AbstractVector, n::Int)
		P = similar(x, n, n)
		return unpack_parameters!(P, x, n)
	end
	
	"""
	    pack_parameters!(dest::AbstractVector, P::AbstractMatrix)
	
	Packs a matrix `P` into a pre-allocated vector `dest`.
	"""
	function pack_parameters!(dest::AbstractVector, P::AbstractMatrix)
	    n = size(P, 1)
	    copyto!(dest, reshape(P, n*n))
	    return dest
	end
	
	"""
	    pack_parameters(P::AbstractMatrix)
	
	Packs a matrix `P` into a flattened vector representation.
	"""
	function pack_parameters(P::AbstractMatrix)
	    n = size(P, 1)
	    dest = similar(P, n*n)  # Allocate once
	    return pack_parameters!(dest, P)
	end
	
	struct QuadraticSymmetryObjective{T<:Real} <: ObjectiveFunction
		    A::Matrix{T}
		    c::Vector{T}
		    n::Int
		    
		    # Pre-allocated buffers for computations
		    P_buffer::Matrix{T}
		    AP_buffer::Matrix{T}
		    APA_buffer::Matrix{T}
		    APAT_buffer::Matrix{T}

			H::SparseMatrixCSC{Float64, Int64}
		    
		    function QuadraticSymmetryObjective(
				A::AbstractMatrix{T},
				c::AbstractVector{T}
			) where {T<:Real}
				
		        n = size(A, 1)
		        if n != size(A, 2)
		            throw(DimensionMismatch("Adjacency matrix A must be square"))
		        end
		        if length(c) != n
		            throw(DimensionMismatch("Length of c must match the dimension of A"))
		        end
		        
		        # Create buffers
		        P_buffer = Matrix{T}(undef, n, n)
		        AP_buffer = Matrix{T}(undef, n, n)
		        APA_buffer = Matrix{T}(undef, n, n)
		        APAT_buffer = Matrix{T}(undef, n, n)

				A_sparse = sparse(A)
				H = -2 * kron(A_sparse, A_sparse)
		        
		        return new{T}(
					Matrix{T}(A), Vector{T}(c), n, 
		            P_buffer, AP_buffer, APA_buffer, APAT_buffer,
					H
				)
		    end
		end
	
	"""
		(f::QuadraticSymmetryObjective)(x::AbstractVector)
	
	Evaluates the objective function at the given point `x`.
	"""
	function (f::QuadraticSymmetryObjective)(x::AbstractVector{T}) where {T<:Real}
		n = f.n
		P = unpack_parameters!(f.P_buffer, x, n)
		
		# A*P
		mul!(f.AP_buffer, f.A, P)
		
		# A*P*A
		mul!(f.APA_buffer, f.AP_buffer, f.A)
		
		# A*P*A*P^T
		mul!(f.APAT_buffer, f.APA_buffer, transpose(P))
		
		# trace(A*P*A*P^T)
		trace_term = tr(f.APAT_buffer)
		
		# Calculate penalty term trace(diag(c)*P)
		penalty_term = sum(f.c[i] * P[i, i] for i in 1:n)
		
		# Return negative of objective (since we're minimizing)
		return -(trace_term - penalty_term)
	end


	function jacobian!(
		grad::AbstractVector,
		f::QuadraticSymmetryObjective,
		x::AbstractVector{T}
	) where {T<:Real}
	    n = f.n
	    P = unpack_parameters!(f.P_buffer, x, n)
	    
	    # Compute A*P
	    mul!(f.AP_buffer, f.A, P)
	    
	    # Compute A*P*A since A is symmetric
	    mul!(f.APA_buffer, f.AP_buffer, f.A)
	    
	    # The gradient is -2*A*P*A + diag(c)
	    # Convert this to flattened form for our optimization variables
	    @inbounds for i in 1:n
	        for j in 1:n
	            idx = (i-1)*n + j
	            
	            # -2*A*P*A term
	            grad[idx] = -2 * f.APA_buffer[i, j]
	            
	            # Add penalty term for diagonal elements
	            if i == j
	                grad[idx] += f.c[i]
	            end
	        end
	    end
	    
	    return grad
	end

	function hess_prototype(f::QuadraticSymmetryObjective)
		A_sparse = sparse(f.A)
		kron(A_sparse, A_sparse)
	end

	function hessian!(
		H, f::QuadraticSymmetryObjective, ::AbstractVector{T}
	) where {T<:Real}
		copy!(H, f.H)
	end
end

# ╔═╡ b85b0b95-c733-4363-ae98-e5e5ae76f7c8
obj = let
	c = 0 * ones(n)
	QuadraticSymmetryObjective(A, c)
end

# ╔═╡ 3411603b-ce02-4f96-ba0a-750dcc09b77d
I[1:n, 1:n] |> pack_parameters |> obj

# ╔═╡ 07b68c47-eccd-434c-99c3-bb5c7f4e9405
function f(A, x)
	n = size(A, 1)
	P = reshape(x, n, n)
	-tr(A * P * transpose(A) * transpose(P))
end

# ╔═╡ 550fc11a-2885-43cf-9dd0-2b3e60edc270
X = reshape(1:9, 3, 3)

# ╔═╡ 717702fa-2d7c-4b76-9c3a-be1c8702cb3a
X |> pack_parameters

# ╔═╡ 6c4ee078-efc6-4776-b3c3-aa2283d1bae7
begin
	Y = zeros(3, 3)
	x = X |> pack_parameters
	unpack_parameters!(Y, x, 3)
end

# ╔═╡ c6dc4dab-7831-4ddd-929c-45e325f2106c
reshape(x, 3, 3)

# ╔═╡ 3303c079-e86a-481b-ace7-e7d9504afd06
reshape(X, 3 * 3)

# ╔═╡ d5bf49c7-3d03-4067-bc89-c882fc76245d
A_sparse = sparse(A)

# ╔═╡ c9421f9a-da81-4160-b4b8-e0611759eae7
kron(A_sparse, A_sparse)

# ╔═╡ 2277b0c2-0ed0-406c-b818-948e70a91cef
reshape([ 1 2; 3 4] |> pack_parameters, 2, 2)

# ╔═╡ a31b9c6d-7b41-497a-984c-460f15e604f2
methods(hessian)

# ╔═╡ 93d98c52-13e1-462a-af2f-fec3b3d7544b
hess_prototype(obj) |> findnz

# ╔═╡ 6c361ee4-e9b3-41c6-9d2b-e9b42226f435
methods(to_permutation_matrix)

# ╔═╡ 5811abf9-8d26-416c-a389-9e1cc70bea2a
E(P, A) = 1/4 * norm(A - P' * A * P)

# ╔═╡ b07cb1d5-fddc-436f-9bea-4c81a2bc1b5e
#solve_qsa_autodiff(A, 0.2; max_iter=1000)

# ╔═╡ 0f7e8087-7d12-4ae6-ae33-7dbf0042b8f6
function project_to_permutations(D::AbstractMatrix) :: AbstractMatrix{Bool}
	Hungarian.munkres(-D) .== 2
end

# ╔═╡ 15223ef8-0235-4a14-9c61-45aae5cfd016


# ╔═╡ 46bd10c4-2038-4786-af34-9579fb3ab58b
"""
	sinkhorn(P::AbstractMatrix)

Projects matrix P to the nearest doubly stochastic matrix using Sinkhorn's algorithm.
"""
function sinkhorn(P::AbstractMatrix{T}; max_iter=1000, tol=1e-8) where {T<:Real}
	n = size(P, 1)
	P_ds = copy(P)
	
	# Ensure non-negativity
	P_ds[P_ds .< 0] .= 0
	
	# Apply Sinkhorn's algorithm
	for iter in 1:max_iter
		# Normalize rows
		row_sums = sum(P_ds, dims=2)
		for i in 1:n
			if row_sums[i] > tol
				P_ds[i, :] ./= row_sums[i]
			else
				P_ds[i, :] .= 1/n
			end
		end
		
		# Normalize columns
		col_sums = sum(P_ds, dims=1)
		for j in 1:n
			if col_sums[j] > tol
				P_ds[:, j] ./= col_sums[j]
			else
				P_ds[:, j] .= 1/n
			end
		end
		
		# Check convergence
		row_error = maximum(abs.(sum(P_ds, dims=2) .- 1))
		col_error = maximum(abs.(sum(P_ds, dims=1) .- 1))
		
		if max(row_error, col_error) < tol
			break
		end
	end
	
	return P_ds
end

# ╔═╡ 619fab72-3598-4671-991e-9d839f97d33a
d = rand(n, n) |> sinkhorn |> pack_parameters

# ╔═╡ 56d68676-8d15-40f5-b7aa-2b496c5cc94d
obj(d)

# ╔═╡ 4973d2f6-b100-4ba5-83bd-d8d346960a1d
grad_approx = Zygote.gradient(x -> f(A, x), d)[1]

# ╔═╡ dd01899f-f044-462b-b806-4c8b09acadc2
grad_exact = let
	grad = Vector{Float64}(undef, n * n)
	jacobian!(grad, obj, d)
end

# ╔═╡ 66241a22-71b9-4a8e-9d4a-5cb638f23f21
grad_approx - grad_exact .|> abs |> sum

# ╔═╡ f2ce75b9-a527-49c9-8b90-f4a8d8924f10
Zygote.hessian(x -> f(A, x), d) |> sparse

# ╔═╡ 2c75615a-a480-47d8-9374-b6fb74ca7062
begin
	H = ones(n * n, n * n)
	hessian!(H, obj, d)
	H |> sparse
end

# ╔═╡ fa510768-38a1-4596-a638-e17d6d6ddb32
"""
	solve_qsa(A::AbstractMatrix, penalty::Real=0.1; max_iter=1000, tol=1e-6)

Solves the approximate symmetry problem using the QSA method.

# Arguments
- `A::AbstractMatrix`: Adjacency matrix of the input graph
- `penalty::Real=0.1`: Penalty coefficient
- `max_iter::Integer=1000`: Maximum number of iterations
- `tol::Real=1e-6`: Convergence tolerance
"""
function solve_qsa(
	A::AbstractMatrix{T},
	penalty::Real=0.1;
	max_iter=1000,
	tol=1e-6
) where {T<:Real}
	
	n = size(A, 1)
	
	# Create penalty vector
	c = fill(convert(T, penalty), n)
	
	# Create objective function
	obj_func = QuadraticSymmetryObjective(A, c)
	
	# Initialize with identity permutation
	P_init = rand(T, n, n) |> sinkhorn
	x0 = pack_parameters(P_init)
	
	# Setup row and column sum constraints
	function cons!(res, x, p)
	    P = reshape(x, n, n)
	    
	    # Row sum constraints
	    for i in 1:n
	        res[i] = sum(P[i, :]) - 1
	    end
	    
	    # Column sum constraints
	    for j in 1:n
	        res[n + j] = sum(P[:, j]) - 1
	    end
	end

	# Create constraint Jacobian prototype with proper sparsity pattern
	function create_constraint_jacobian(n::Int, T::Type=Float64)
	    n_vars = n*n
	    n_constraints = 2*n
	    
	    # Pre-allocate arrays for sparse matrix construction
	    # Each constraint has exactly n non-zero entries
	    nnz = 2*n*n  # Total non-zeros
	    I = Vector{Int}(undef, nnz)
	    J = Vector{Int}(undef, nnz)
	    V = ones(T, nnz)  # All entries are 1.0
	    
	    idx = 1
	    
	    # Row sum constraints
		for i in 1:n
	        for j in 1:n
	            # Constraint n+i depends on variables (j-1)*n + i
	            I[idx] = i
	            J[idx] = (j-1)*n + i
	            idx += 1
	        end
	    end
	    
	    # Column sum constraints
		for i in 1:n
	        for j in 1:n
	            # Constraint i depends on variables (i-1)*n + j
	            I[idx] = n + i
	            J[idx] = (i-1)*n + j
	            idx += 1
	        end
	    end
	    
	    return sparse(I, J, V, n_constraints, n_vars)
	end
	
	cons_jac_prototype = create_constraint_jacobian(n, T)
	function cons_j!(J, x, p)
	    # For a sparse Jacobian with constant pattern, we don't need to do anything
	    # The structure is already defined by the prototype
	    # If J is a dense matrix, we'd need to copy from the sparse prototype
	    if J isa SparseMatrixCSC
	        fill!(nonzeros(J), one(eltype(J)))
	    else
	        # Copy from sparse to dense (less efficient but handles any matrix type)
	        copyto!(J, cons_jac_prototype)
	    end
	    return nothing
	end


	# Create constraint Hessian prototype (all zeros since constraints are linear)
	cons_hess_prototype = [spzeros(T, n*n, n*n) for _ in 1:2*n]
	function cons_h!(H, x, p)
		# Do nothing since the hessian is empty
	    return nothing
	end
	
	# Create optimization problem
	optf = OptimizationFunction(
		(x, p) -> obj_func(x);
		grad=(g, x, p) -> jacobian!(g, obj_func, x),
		hess=(H, x, p) -> hessian!(H, obj_func, x),
		hess_prototype=hess_prototype(obj_func),
		cons=cons!,
   		cons_j=cons_j!,
		cons_jac_prototype=cons_jac_prototype,
		cons_h=cons_h!,
		cons_hess_prototype=cons_hess_prototype,
	)
	
	# Setup bounds
	lb = zeros(n*n)
	ub = fill(Inf, n*n)
	
	# Problem constraints
	lcons = zeros(T, 2*n)
	ucons = zeros(T, 2*n)
	
	prob = OptimizationProblem(
	    optf,
	    x0,
	    nothing;
	    lb=lb,
	    ub=ub,
	    lcons=lcons,
	    ucons=ucons
	)
	
	# Setup Ipopt solver
	optimizer = OptimizationMOI.MOI.OptimizerWithAttributes(
		Ipopt.Optimizer,
		"max_iter" => max_iter,
		"tol" => tol,
		
		"hessian_constant" => "yes",
    	"jac_c_constant" => "yes",
    	"jac_d_constant" => "yes",
    	
		"num_linear_variables" => n*n,
		"print_level" => 3
	)
	
	# Solve the problem
	start_time = time()
	sol = SciMLBase.solve(prob, optimizer)
	solve_time = time() - start_time
	
	# Reshape solution to matrix form
	P_sol = reshape(sol.u, n, n)
	
	# Project to permutation matrix
	P_perm = project_to_permutations(P_sol)
	
	# Calculate objective value with the permutation matrix
	final_obj = obj_func(pack_parameters(P_perm))
	
	return P_perm, P_sol, final_obj, solve_time
end
	

# ╔═╡ 5f0c5ab7-1513-42d1-a052-2f0c69ce5dcc
P_perm, P_sol, _, _ = solve_qsa(A, 0.1; max_iter=1000)

# ╔═╡ 267e2209-d08a-4ae8-bf63-88924f0394f3
plot(
	heatmap(P_sol),
	heatmap(P_perm);
	aspect_ratio=1
)

# ╔═╡ 2e859119-a60a-4a27-b468-b690ca40724d
lambdas, factors = let
	D = copy(P_sol)
	factors = Matrix{Bool}[]
	lambdas = Float64[]

	for i in 1:10
		F = D |> project_to_permutations
		lambda = D[F] |> minimum

		D .-= lambda * F
		
		push!(factors, F)
		push!(lambdas, lambda)
	end

	lambdas, factors
end

# ╔═╡ a7a8d924-888e-4d47-a202-f6eec2c0af30
plot(
	heatmap(factors[1]),
	heatmap(factors[2]);
	aspect_ratio=1
)

# ╔═╡ ca655f10-d4f1-47c0-9cc4-d7aa9baaaa21
begin
	results = Dict(
		"n"=>[],
		"c"=>[],
		"run"=>[],
		"E"=>[],
		"time"=>[],
	)
	cs = Base.LogRange(0.001, 5, 10)
	nruns = 10

	for c in cs
		local run = 1
		@info "Choosing penalty c=$c"
		while run <= nruns
			@info "Solving run $run"
			P_perm, P_sol, _, time = solve_qsa(A, c; max_iter=1000)
			
			if tr(P_perm) == n
				@info "Solution is an identity, retrying..."
				continue
			else
				@info tr(P_perm)
			end
			
			push!(results["n"], n)
			push!(results["c"], c)
			push!(results["run"], run)
			push!(results["E"], E(P_perm, A))
			push!(results["time"], time)
			
			run += 1
		end
	end

	df = DataFrame(results)
end

# ╔═╡ fdf79e60-bbe2-417e-8003-e23081fc7976
begin
	
	# Create a two-panel plot with distributions
	p = plot(
	    layout = (1, 2),
	    size = (800, 400),
	    legend = false,
	)
	
	# First subplot: Distribution of E values
	histogram!(p[1], df.E, 
	    bins = 15,
	    normalize = true,
	    title = "Distribution of E",
	    xlabel = "E values",
	    ylabel = "Frequency",
	    fillalpha = 0.6,
	    color = :blue
	)
	
	# Optional: Add density curve
	density!(p[1], df.E, color = :red, linewidth = 2)
	
	# Second subplot: Distribution of time values
	histogram!(p[2], df.time, 
	    bins = 15,
	    normalize = true,
	    title = "Distribution of Execution Time",
	    xlabel = "Time (s)",
	    ylabel = "Frequency",
	    fillalpha = 0.6,
	    color = :green
	)
	
	# Optional: Add density curve
	density!(p[2], df.time, color = :red, linewidth = 2)
	
	# Set plot title
	plot!(p, plot_title = "Algorithm Performance Metrics")
	
end

# ╔═╡ 69b24df3-8cd1-43d1-9d6a-8fa975b1bd11
begin
	plot(
		violin(["E"], df.E, label="E(A, P)"),
		violin(["Time"], df.time,  label="Solution Time")
	)
end

# ╔═╡ 649c700b-a0a0-401e-8ec1-326cf75daf6f
boxplot(string.(round.(df.c, digits=3)), df.E,
	title = "E Distribution by Penalty Parameter",
	xlabel = "c value",
	ylabel = "E",
	legend = false,
	color = :lightblue,
	whisker_width = 0.5,
	size = (800, 500)
)

# ╔═╡ 17b7d076-2263-44d8-baac-a07f2562dc58
violin(string.(round.(df.c, digits=3)), df.E,
	title = "E Distribution by Penalty Parameter",
	xlabel = "c value",
	ylabel = "E",
	legend = false,
	color = :lightblue,
	alpha = 0.7,
	size = (800, 500)
)

# ╔═╡ 5cfd0dd7-3f51-41d3-a0d7-560ff2437223
begin
	grouped_data = combine(groupby(df, :c), 
		:E => mean => :mean_E,
		:E => std => :std_E)
	
	scatter(grouped_data.c, grouped_data.mean_E,
		yerror = grouped_data.std_E,
		title = "Mean E by Penalty Parameter",
		xlabel = "c value (log scale)",
		ylabel = "Mean E",
		legend = false,
		markersize = 6,
		color = :blue,
		xscale = :log10,  # Using log scale for c
		size = (800, 500)
	)
end

# ╔═╡ 0e207c13-5dc3-4f9d-be1f-ceedd060d796
begin
	p4 = plot(layout = (2, 1), size = (900, 800))
	
	# Top panel: Scatter with error bars
	scatter!(p4[1], grouped_data.c, grouped_data.mean_E,
	    yerror = grouped_data.std_E,
	    title = "Mean E by Penalty Parameter",
	    xlabel = "",  # No label here since we have the bottom plot
	    ylabel = "E",
	    legend = false,
	    markersize = 6,
	    color = :blue,
	    xscale = :log10
	)
	
	# Bottom panel: Heatmap-style visualization
	# Convert to log space for c values if they span multiple orders of magnitude
	df.log_c = log10.(df.c)
	bins = range(minimum(df.log_c), maximum(df.log_c), length=10)
	bin_indices = [findlast(bins .<= val) for val in df.log_c]
	
	# Create grouped boxplots or violin plots
	violin!(p4[2], bin_indices, df.E,
	    xticks = (1:length(bins), string.(round.(10 .^ bins, digits=2))),
	    title = "Energy Distribution by Penalty Parameter",
	    xlabel = "c value (log scale)",
	    ylabel = "Energy (E)",
	    legend = false,
	    color = :lightblue
	)
end

# ╔═╡ dc9a6c0f-156c-40e4-8f32-30b86a903b3a
begin
	"""
	    QSAMethod <: AbstractMethod
	
	Implementation of the Quadratic Symmetry Approximator (QSA) method.
	
	# Fields
	- `name::String`: Name of the method
	- `version::String`: Version identifier
	- `parameters::Dict{Symbol,Any}`: Parameter dictionary
	"""
	struct QSAMethod <: AbstractMethod
	    name::String
	    version::String
	    parameters::Dict{Symbol,Any}
	end
	
	"""
	    QSAMethod(; name="QSA", version="v1", penalty=0.1, max_iter=1000, tol=1e-6)
	
	Construct a QSAMethod with specified parameters.
	"""
	function QSAMethod(; 
	    name::String="QSA", 
	    version::String="v1",
	    penalty::Number=0.1,
	    max_iter::Integer=1000,
	    tol::Real=1e-6
	)
	    parameters = Dict{Symbol,Any}(
	        :penalty => penalty,
	        :max_iter => max_iter,
	        :tol => tol
	    )
	    
	    return QSAMethod(name, version, parameters)
	end
	
	"""
	    supported_metrics(::QSAMethod)
	
	Return metrics supported by the QSA method.
	"""
	function supported_metrics(::QSAMethod)
	    return Symbol[:time, :s_metric, :objective_value, :convergence]
	end
	
	"""
	    get_parameter(method::QSAMethod, key::Symbol, default=nothing)
	
	Get a parameter value with an optional default.
	"""
	function get_parameter(method::QSAMethod, key::Symbol, default=nothing)
	    return get(method.parameters, key, default)
	end
	
	"""
	    set_parameter!(method::QSAMethod, key::Symbol, value)
	
	Set a parameter value.
	"""
	function set_parameter!(method::QSAMethod, key::Symbol, value)
	    method.parameters[key] = value
	    return method
	end
	
	"""
	    solve(method::QSAMethod, instance::AbstractInstance)
	
	Apply the QSA method to find an approximate symmetry.
	"""
	function solve(method::QSAMethod, instance::AbstractInstance)
	    # Get parameters
	    penalty = get_parameter(method, :penalty, 0.1)
	    max_iter = get_parameter(method, :max_iter, 1000)
	    tol = get_parameter(method, :tol, 1e-6)
	    
	    # Get adjacency matrix
	    A = Instances.adjacency(instance)
	    
	    # Suppress Ipopt output
	    original_stdout = stdout
	    rd, = redirect_stdout()
	    
	    # Solve the QSA problem
	    P_perm, P_doubly, obj_value, solve_time = solve_qsa(
	        A, 
	        penalty;
	        max_iter=max_iter,
	        tol=tol
	    )
	    
	    # Restore stdout and capture Ipopt output
	    redirect_stdout(original_stdout)
	    closewrite(rd)
	    solver_output = String(readavailable(rd))
	    close(rd)
	    
	    # Create solution object
	    solution = Solution(P_perm, instance)
	    
	    # Add metrics
	    set_metric!(solution, :time, solve_time)
	    set_metric!(solution, :objective_value, obj_value)
	    set_metric!(solution, :doubly_stochastic, P_doubly)
	    set_metric!(solution, :stdout, solver_output)
	    
	    # Calculate S metric if not already present
	    if !has_metric(solution, :s_metric)
	        s_metric = calculate_s_metric(A, P_perm)
	        set_metric!(solution, :s_metric, s_metric)
	    end
	    
	    return solution
	end
	
	"""
	    calculate_s_metric(A::AbstractMatrix, P::AbstractMatrix)
	
	Calculate the S metric for a permutation matrix P and adjacency matrix A.
	"""
	function calculate_s_metric(A::AbstractMatrix, P::AbstractMatrix)
	    n = size(A, 1)
	    norm_factor = n * (n - 1)
	    
	    # Calculate A - P*A*P'
	    PA = P * A
	    PAP = PA * transpose(P)
	    diff = A - PAP
	    
	    # Calculate Frobenius norm
	    s_value = norm(diff) / norm_factor
	    
	    return s_value
	end
end

# ╔═╡ 961ca64d-d0a1-4f91-88f9-743fb81c6fa3
let
	R = rand(n, n)
	D = R |> sinkhorn
	p_D = D |> heatmap

	P = D |> project_to_permutations
	p_P = P |> heatmap

	plot(p_D, p_P)
end

# ╔═╡ fa43a44b-dc84-42cb-82f4-fd9905044a73
function solve_qsa_autodiff(
	A::AbstractMatrix{T},
	penalty::Real=0.1;
	max_iter=1000,
	tol=1e-6
) where {T<:Real}
	
	n = size(A, 1)
	
	# Create penalty vector
	c = fill(convert(T, penalty), n)
	
	# Initialize with identity permutation
	P_init = rand(T, n, n) |> sinkhorn
	x0 = pack_parameters(P_init)
	
	# Setup row and column sum constraints
	function cons!(res, x, p)
	    P = reshape(x, n, n)
	    
	    # Row sum constraints
	    for i in 1:n
	        res[i] = sum(P[i, :]) - 1
	    end
	    
	    # Column sum constraints
	    for j in 1:n
	        res[n + j] = sum(P[:, j]) - 1
	    end
	end

	function objective(x)
		P = reshape(x, n, n)
		-tr(A * P * transpose(A) * transpose(P))
	end

	# Create optimization problem
	optf = OptimizationFunction(
		(x, p) -> objective(x),
		Optimization.AutoForwardDiff();
		cons=cons!,
	)
	
	# Setup bounds
	lb = zeros(n*n)
	ub = fill(Inf, n*n)
	
	# Problem constraints
	lcons = zeros(T, 2*n)
	ucons = zeros(T, 2*n)
	
	prob = OptimizationProblem(
	    optf,
	    x0,
	    nothing;
	    lb=lb,
	    ub=ub,
	    lcons=lcons,
	    ucons=ucons
	)
	
	# Setup Ipopt solver
	optimizer = OptimizationMOI.MOI.OptimizerWithAttributes(
		Ipopt.Optimizer,
		"max_iter" => max_iter,
		"tol" => tol,

		"hessian_constant" => "yes",
    	"jac_c_constant" => "yes",
    	"jac_d_constant" => "yes",
    	
		"num_linear_variables" => n*n,
		"print_level" => 3
	)
	
	# Solve the problem
	start_time = time()
	sol = SciMLBase.solve(prob, optimizer)
	solve_time = time() - start_time
	
	# Reshape solution to matrix form
	P_sol = reshape(sol.u, n, n)
	
	# Project to permutation matrix
	P_perm = project_to_permutations(P_sol)
	
	# Calculate objective value with the permutation matrix
	final_obj = objective(pack_parameters(P_perm))
	
	return P_perm, P_sol, final_obj, solve_time
end
	

# ╔═╡ Cell order:
# ╠═95f44e1a-1644-11f0-3029-5f3b87f55957
# ╠═18c9368b-1964-452b-a9fb-4bd716c327f5
# ╠═a62c3acd-c47c-48fa-99fd-da9000c4a232
# ╠═16f8ace3-042c-4d45-9c2a-ad539bea6d13
# ╠═e7cc1995-38aa-4304-b8e1-46e31b86ffd5
# ╠═0e09b4c0-4006-43f5-9160-ef41310e11e1
# ╠═9cb38998-a485-41f4-9dd5-35759875cf22
# ╠═5cfb8286-0975-40a8-96e1-5c5d7b42a7d5
# ╠═b6a1cc10-3b3f-4d23-85dc-4c31f0420e7c
# ╠═b85b0b95-c733-4363-ae98-e5e5ae76f7c8
# ╠═619fab72-3598-4671-991e-9d839f97d33a
# ╠═3411603b-ce02-4f96-ba0a-750dcc09b77d
# ╠═56d68676-8d15-40f5-b7aa-2b496c5cc94d
# ╠═07b68c47-eccd-434c-99c3-bb5c7f4e9405
# ╠═8daedc07-df00-46ad-835b-d6fc70d89575
# ╠═4973d2f6-b100-4ba5-83bd-d8d346960a1d
# ╠═dd01899f-f044-462b-b806-4c8b09acadc2
# ╠═550fc11a-2885-43cf-9dd0-2b3e60edc270
# ╠═717702fa-2d7c-4b76-9c3a-be1c8702cb3a
# ╠═6c4ee078-efc6-4776-b3c3-aa2283d1bae7
# ╠═c6dc4dab-7831-4ddd-929c-45e325f2106c
# ╠═3303c079-e86a-481b-ace7-e7d9504afd06
# ╠═66241a22-71b9-4a8e-9d4a-5cb638f23f21
# ╠═f2ce75b9-a527-49c9-8b90-f4a8d8924f10
# ╠═2c75615a-a480-47d8-9374-b6fb74ca7062
# ╠═fa510768-38a1-4596-a638-e17d6d6ddb32
# ╠═961ca64d-d0a1-4f91-88f9-743fb81c6fa3
# ╠═d5bf49c7-3d03-4067-bc89-c882fc76245d
# ╠═c9421f9a-da81-4160-b4b8-e0611759eae7
# ╠═2277b0c2-0ed0-406c-b818-948e70a91cef
# ╠═a31b9c6d-7b41-497a-984c-460f15e604f2
# ╠═93d98c52-13e1-462a-af2f-fec3b3d7544b
# ╠═6c361ee4-e9b3-41c6-9d2b-e9b42226f435
# ╠═5f0c5ab7-1513-42d1-a052-2f0c69ce5dcc
# ╠═267e2209-d08a-4ae8-bf63-88924f0394f3
# ╠═2e859119-a60a-4a27-b468-b690ca40724d
# ╠═a7a8d924-888e-4d47-a202-f6eec2c0af30
# ╠═5811abf9-8d26-416c-a389-9e1cc70bea2a
# ╠═ca655f10-d4f1-47c0-9cc4-d7aa9baaaa21
# ╠═fdf79e60-bbe2-417e-8003-e23081fc7976
# ╠═69b24df3-8cd1-43d1-9d6a-8fa975b1bd11
# ╠═649c700b-a0a0-401e-8ec1-326cf75daf6f
# ╠═17b7d076-2263-44d8-baac-a07f2562dc58
# ╠═5cfd0dd7-3f51-41d3-a0d7-560ff2437223
# ╠═0e207c13-5dc3-4f9d-be1f-ceedd060d796
# ╠═fa43a44b-dc84-42cb-82f4-fd9905044a73
# ╠═b07cb1d5-fddc-436f-9bea-4c81a2bc1b5e
# ╠═0f7e8087-7d12-4ae6-ae33-7dbf0042b8f6
# ╠═15223ef8-0235-4a14-9c61-45aae5cfd016
# ╠═dc9a6c0f-156c-40e4-8f32-30b86a903b3a
# ╠═46bd10c4-2038-4786-af34-9579fb3ab58b
