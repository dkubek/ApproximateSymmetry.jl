using SparseArrays
using LinearAlgebra
using Optimization
using OptimizationMOI
using Ipopt

"""
    PermutationOptimizationMethod <: AbstractMethod

A method for computing approximate graph symmetries using continuous optimization with
analytical derivatives.

# Fields
- `name::String`: Name of the method
- `version::String`: Version identifier
- `parameters::Dict{Symbol, Any}`: Additional parameters
"""
struct PermutationOptimizationMethod <: AbstractMethod
    name::String
    version::String
    parameters::Dict{Symbol,Any}
end

"""
    PermutationOptimizationMethod(;name="PermOpt", version="v1", c=0.1, x0=0.25, max_iter=1000, tol=1e-8)

Construct a PermutationOptimizationMethod with the given parameters.

# Arguments
- `name::String="PermOpt"`: Name of the method
- `version::String="v1"`: Version identifier
- `c::Number=0.1`: Weight parameter for diagonal elements
- `x0::Number=0.25`: Initial value for all elements
- `max_iter::Integer=1000`: Maximum number of iterations
- `tol::Real=1e-8`: Convergence tolerance
"""
function PermutationOptimizationMethod(;
    name::String="PermOpt",
    version::String="v1",
    c::Number=0.1,
    x0::Number=0.25,
    max_iter::Integer=1000,
    tol::Real=1e-8
)
    parameters = Dict{Symbol,Any}(
        :c => c,
        :x0 => x0,
        :max_iter => max_iter,
        :tol => tol
    )
    return PermutationOptimizationMethod(name, version, parameters)
end

"""
    supported_metrics(method::PermutationOptimizationMethod)

Get all metrics supported by this method.
"""
function supported_metrics(::PermutationOptimizationMethod)
    return Symbol[:time, :s_metric, :frobenius_norm, :is_permutation, :perm_type, :objective_value]
end

"""
    validate_matrix(A::AbstractMatrix{T}) where T<:Real

Check if the matrix A meets the required properties:
- Symmetric
- Zeros on diagonal
"""
function validate_matrix(A::AbstractMatrix{T}) where {T<:Real}
    # Check symmetry
    if !issymmetric(A)
        @warn "Matrix is not symmetric"
        return false
    end

    # Check diagonal (optional, can be relaxed)
    for i in 1:size(A, 1)
        if A[i, i] != zero(T)
            @warn "Matrix has non-zeros on diagonal"
            return false
        end
    end

    return true
end

"""
    objective_function(X::AbstractVector{T}, params) where T<:Real

Calculate the objective value: -trace(A*X*A*X' - c*X)
"""
function objective_function(X::AbstractVector{T}, params) where {T<:Real}
    A = params.A
    c = params.c
    n = size(A, 1)
    X_mat = reshape(X, n, n)

    # Calculate A*X
    AX = A * X_mat

    # Calculate (A*X) * (A*X)'
    AXAX = AX * transpose(AX)

    # Calculate trace
    trace_term = tr(AXAX)

    # If c > 0, add diagonal term
    diag_term = zero(T)
    if c > zero(T)
        diag_term = c * sum(diag(X_mat))
    end

    # Return negative objective (we're minimizing)
    return -(trace_term - diag_term)
end

"""
    objective_gradient!(G::AbstractVector{T}, X::AbstractVector{T}, params) where T<:Real

Calculate the gradient of objective function analytically.
"""
function objective_gradient!(G::AbstractVector{T}, X::AbstractVector{T}, params) where {T<:Real}
    A = params.A
    c = params.c
    n = size(A, 1)
    X_mat = reshape(X, n, n)

    # Calculate A*X and X*A
    AX = A * X_mat
    XA = X_mat * A

    # Calculate the gradient analytically
    # ∇f(X) = -2*(A'*A*X*A' + A*X*A*A') + c*I (where I only applies to diagonal)

    # First part: A'*A*X*A'
    part1 = transpose(A) * AX * transpose(A)

    # Second part: A*X*A*A'
    part2 = A * XA * A

    # Combine with the negative sign
    grad_mat = -2.0 * (part1 + part2)

    # Add diagonal term if c > 0
    if c > zero(T)
        for i in 1:n
            grad_mat[i, i] += c
        end
    end

    # Flatten gradient to vector form
    copyto!(G, vec(grad_mat))
end

"""
    objective_hessian!(H::AbstractMatrix{T}, X::AbstractVector{T}, params) where T<:Real

Calculate the Hessian of objective function analytically.
"""
function objective_hessian!(H::AbstractMatrix{T}, X::AbstractVector{T}, params) where {T<:Real}
    n = size(params.A, 1)
    n_vars = n^2

    # Reset Hessian to zeros
    fill!(H, zero(T))

    # Set all off-diagonal elements to -2.0
    for i in 1:n_vars
        for j in 1:n_vars
            if i != j
                H[i, j] = -2.0
            end
        end
    end

    # Diagonal elements are zero (already set above)
end

"""
    constraints!(c::AbstractVector{T}, X::AbstractVector{T}, params) where T<:Real

Apply row and column sum constraints: all rows and columns must sum to 1.
"""
function constraints!(c::AbstractVector{T}, X::AbstractVector{T}, params) where {T<:Real}
    n = size(params.A, 1)
    X_mat = reshape(X, n, n)

    # Row sums
    for i in 1:n
        c[i] = sum(view(X_mat, i, :)) - one(T)
    end

    # Column sums
    for j in 1:n
        c[n+j] = sum(view(X_mat, :, j)) - one(T)
    end
end

"""
    constraint_jacobian!(J::AbstractMatrix{T}, X::AbstractVector{T}, params) where T<:Real

Calculate the Jacobian of constraints analytically.
"""
function constraint_jacobian!(J::AbstractMatrix{T}, X::AbstractVector{T}, params) where {T<:Real}
    n = size(params.A, 1)

    # Clear existing values
    fill!(J, zero(T))

    # Row constraints Jacobian
    for i in 1:n
        for j in 1:n
            # For row i, all elements in that row affect the constraint
            J[i, (i-1)*n+j] = one(T)
        end
    end

    # Column constraints Jacobian
    for i in 1:n
        for j in 1:n
            # For column i, all elements in that column affect the constraint
            J[n+i, (j-1)*n+i] = one(T)
        end
    end
end

"""
    optimize_permutation(A::AbstractMatrix{T}; params::Dict{Symbol,Any}) where T<:Real

Find the permutation matrix that best approximates matrix A.
Uses analytical derivatives for efficiency.

# Arguments
- `A`: Input symmetric matrix
- `params`: Dictionary of parameters including:
  - `c`: Weight parameter for diagonal elements (default: 0.1)
  - `x0`: Initial value for all elements (default: 0.25)
  - `max_iter`: Maximum iterations (default: 1000)
  - `tol`: Tolerance for convergence (default: 1e-8)
  - `eps`: Tolerance for permutation check (default: 1e-6)

# Returns
- `X`: The optimized matrix
- `is_permutation`: Whether X is a permutation matrix
- `perm_type`: Classification of the permutation solution
- `objective_value`: The final objective value
"""
function optimize_permutation(A::AbstractMatrix{T}; params::Dict{Symbol,Any}) where {T<:Real}
    validate_matrix(A)

    n = size(A, 1)
    n_vars = n^2

    # Extract parameters with defaults
    c = convert(T, get(params, :c, 0.1))
    x0 = convert(T, get(params, :x0, 0.25))
    max_iter = get(params, :max_iter, 1000)
    tol = convert(T, get(params, :tol, 1e-8))
    eps = convert(T, get(params, :eps, 1e-6))

    # Initial point
    x_init = fill(x0, n_vars)

    # Number of constraints (row and column sums)
    n_constraints = 2 * n

    # Setup bounds
    lb = zeros(T, n_vars)
    ub = ones(T, n_vars)

    constraint_lb = zeros(T, n_constraints)
    constraint_ub = zeros(T, n_constraints)

    # Sparsity pattern for Hessian (all elements except diagonal)
    hessian_sparsity = trues(n_vars, n_vars)
    for i in 1:n_vars
        hessian_sparsity[i, i] = false
    end

    # Sparsity pattern for Jacobian
    jac_sparsity = falses(n_constraints, n_vars)

    # Row constraints
    for i in 1:n
        for j in 1:n
            jac_sparsity[i, (i-1)*n+j] = true
        end
    end

    # Column constraints
    for i in 1:n
        for j in 1:n
            jac_sparsity[n+i, (j-1)*n+i] = true
        end
    end

    # Create optimization problem with analytical derivatives
    optf = OptimizationFunction(
        objective_function,
        grad=objective_gradient!,
        hess=objective_hessian!,
        cons=constraints!,
        cons_j=constraint_jacobian!,
        hess_prototype=sparse(hessian_sparsity),
        # cons_j_prototype=sparse(jac_sparsity)
    )

    prob = OptimizationProblem(
        optf,
        x_init,
        (A=A, c=c),
        lb=lb,
        ub=ub,
        lcons=constraint_lb,
        ucons=constraint_ub
    )

    # Configure Ipopt solver
    optimizer = OptimizationMOI.MOI.OptimizerWithAttributes(
        Ipopt.Optimizer,
        "max_iter" => max_iter,
        "tol" => tol,
        "print_level" => 0  # Suppress output
    )

    # Solve using Ipopt with analytical derivatives
    sol = SciMLBase.solve(prob, optimizer)

    # Reshape solution to matrix
    X_sol = reshape(sol.u, n, n)

    # Check if solution is a permutation matrix
    is_perm, perm_type = check_permutation_solution(X_sol, eps)

    return X_sol, is_perm, perm_type, sol.objective
end

"""
    check_permutation_solution(X::AbstractMatrix{T}, eps::Real=1e-6) where T<:Real

Check if matrix X approximates a permutation matrix within tolerance eps.

# Returns
- `is_permutation`: Whether X is a permutation matrix
- `permutation_type`: Classification of the permutation ("PERM1", "PERM2", or "FINAL")
"""
function check_permutation_solution(X::AbstractMatrix{T}, eps::Real=1e-6) where {T<:Real}
    n = size(X, 1)

    # Strong criterion: elements are either ≈0 or ≈1
    strong_perm = true
    for x in X
        if !(x < eps || x > one(T) - eps)
            strong_perm = false
            break
        end
    end

    # Check if there's exactly one "1" in each row and column
    if strong_perm
        col_counts = zeros(Int, n)
        row_counts = zeros(Int, n)

        for i in 1:n
            for j in 1:n
                if X[i, j] > one(T) - eps
                    row_counts[i] += 1
                    col_counts[j] += 1
                end
            end
        end

        if all(==(1), row_counts) && all(==(1), col_counts)
            return true, "PERM1"
        else
            return false, "FINAL"
        end
    end

    # Weaker criterion with larger eps
    weak_eps = convert(T, 0.1)
    weak_perm = true
    for x in X
        if !(x < weak_eps || x > one(T) - weak_eps)
            weak_perm = false
            break
        end
    end

    if weak_perm
        col_counts = zeros(Int, n)
        row_counts = zeros(Int, n)

        for i in 1:n
            for j in 1:n
                if X[i, j] > one(T) - weak_eps
                    row_counts[i] += 1
                    col_counts[j] += 1
                end
            end
        end

        if all(==(1), row_counts) && all(==(1), col_counts)
            return true, "PERM2"
        end
    end

    return false, "FINAL"
end

"""
    calculate_frobenius_norm(X::AbstractMatrix, A::AbstractMatrix)

Calculate the Frobenius norm between A and X*A*X'.
"""
function calculate_frobenius_norm(X::AbstractMatrix, A::AbstractMatrix)
    # X*A*X' - A
    diff = X * A * transpose(X) - A

    # Frobenius norm
    return norm(diff, 2)
end

"""
    solve(method::PermutationOptimizationMethod, instance::Instances.MatrixInstance)

Apply the PermutationOptimization method to a matrix instance.
Returns a Solution object containing the result and computed metrics.
"""
function solve(method::PermutationOptimizationMethod, instance::AbstractInstance)
    # Start timing

    # Extract parameters
    params = Dict{Symbol,Any}(
        :c => get(method.parameters, :c, 0.1),
        :x0 => get(method.parameters, :x0, 0.25),
        :max_iter => get(method.parameters, :max_iter, 1000),
        :tol => get(method.parameters, :tol, 1e-8),
        :eps => get(method.parameters, :eps, 1e-6)
    )

    # Get matrix from instance
    A = adjacency(A)

    start_time = time()

    # Run optimization
    P, is_perm, perm_type, objective_value = optimize_permutation(A, params=params)

    # End timing
    solve_time = time() - start_time

    # Create solution object
    solution = Solution(P, instance)

    # Add metrics
    set_metric!(solution, :time, solve_time)
    set_metric!(solution, :is_permutation, is_perm)
    set_metric!(solution, :perm_type, perm_type)
    set_metric!(solution, :objective_value, objective_value)

    # Calculate Frobenius norm if result is a permutation
    if is_perm
        frob_norm = calculate_frobenius_norm(P, A)
        set_metric!(solution, :frobenius_norm, frob_norm)
    end

    # Calculate S(A) metric
    s_value = norm(A - P * A * P', 2) / (instance.n * (instance.n - 1))
    set_metric!(solution, :s_metric, s_value)

    return solution
end

export PermutationOptimizationMethod, optimize_permutation, check_permutation_solution
