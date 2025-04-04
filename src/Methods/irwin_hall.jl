using LinearAlgebra
using SparseArrays
using Optimization
using OptimizationMOI
using Ipopt

"""
    irwin_hall_3(x::T) where T<:Real

Calculate the Irwin-Hall distribution for 3 uniform random variables.
This function is used as a basis for approximating permutation matrices.
"""
function irwin_hall_3(x::T) where {T<:Real}
    if x < zero(T)
        return zero(T)
    elseif x < one(T)
        return convert(T, 0.5) * x^2
    elseif x < convert(T, 2)
        return convert(T, 0.5) * (-2 * x^2 + 6 * x - 3)
    elseif x <= convert(T, 3)
        return convert(T, 0.5) * (3 - x)^2
    else
        return zero(T)
    end
end

"""
    irwin_hall_3_derivative(x::T) where T<:Real

First derivative of the `irwin_hall_3` function.
"""
function irwin_hall_3_derivative(x::T) where {T<:Real}
    if x < zero(T)
        return zero(T)
    elseif x < one(T)
        return x
    elseif x < convert(T, 2)
        return -2 * x + 3
    elseif x <= convert(T, 3)
        return x - 3
    else
        return zero(T)
    end
end

"""
    irwin_hall_3_hessian(x::T) where T<:Real

Second derivative of the `irwin_hall_3` function.
"""
function irwin_hall_3_hessian(x::T) where {T<:Real}
    if x < zero(T)
        return zero(T)
    elseif x < one(T)
        return one(T)
    elseif x < convert(T, 2)
        return -2 * one(T)
    elseif x <= convert(T, 3)
        return one(T)
    else
        return zero(T)
    end
end

"""
    periodic(f::Function, period::Integer)

Create a periodic version of function f with given period.
"""
function periodic(f::Function, period::Integer)
    # Enough for our use case to repeat 3 times
    return x -> f(x + period) + f(x) + f(x - period)
end

"""
    PermutationVectorIHPenalized

A type that implements an optimization approach for approximating graph symmetries
using the Irwin-Hall distribution to construct continuous relaxations of permutation matrices.
"""
struct PermutationVectorIHPenalized{T<:Real,B<:Function,DB<:Function,D2B<:Function} <: ObjectiveFunction
    A::Matrix{T}
    c_mat::Diagonal{T}

    P_buffer::Matrix{T}
    AP_buffer::Matrix{T}
    APA_buffer::Matrix{T}

    b::B
    db::DB
    d2b::D2B

    n::Int
end

# Simple constructor without defaults first
function PermutationVectorIHPenalized(
    A::AbstractMatrix,
    c::AbstractVector;
    T::Type{<:Real}=promote_type(Float64, eltype(A), eltype(c))
)
    n = size(A, 1)
    @assert n == size(c, 1) "Matrix A and vector c must have compatible dimensions"

    # Convert inputs to specified type
    A_mat = convert(Matrix{T}, A)
    c_mat = Diagonal(convert(Vector{T}, c))

    # Create buffers
    P_buffer = Matrix{T}(undef, n, n)
    AP_buffer = Matrix{T}(undef, n, n)
    APA_buffer = Matrix{T}(undef, n, n)

    # Create default functions
    b_func = periodic(x -> irwin_hall_3(x + 3 / 2), n)
    db_func = periodic(x -> irwin_hall_3_derivative(x + 3 / 2), n)
    d2b_func = periodic(x -> irwin_hall_3_hessian(x + 3 / 2), n)

    return PermutationVectorIHPenalized{T,typeof(b_func),typeof(db_func),typeof(d2b_func)}(
        A_mat, c_mat,
        P_buffer, AP_buffer, APA_buffer,
        b_func, db_func, d2b_func,
        n
    )
end

# Constructor with custom functions
function PermutationVectorIHPenalized(
    A::AbstractMatrix,
    c::AbstractVector,
    b::Function,
    db::Function,
    d2b::Function;
    T::Type{<:Real}=promote_type(Float64, eltype(A), eltype(c))
)
    n = size(A, 1)
    @assert n == size(c, 1) "Matrix A and vector c must have compatible dimensions"

    # Convert inputs to specified type
    A_mat = convert(Matrix{T}, A)
    c_mat = Diagonal(convert(Vector{T}, c))

    # Create buffers
    P_buffer = Matrix{T}(undef, n, n)
    AP_buffer = Matrix{T}(undef, n, n)
    APA_buffer = Matrix{T}(undef, n, n)

    return PermutationVectorIHPenalized{T,typeof(b),typeof(db),typeof(d2b)}(
        A_mat, c_mat,
        P_buffer, AP_buffer, APA_buffer,
        b, db, d2b,
        n
    )
end

function create_perm_matrix!(
    P::AbstractMatrix{T},
    f::PermutationVectorIHPenalized{T},
    x::AbstractVector{TX}
) where {T<:Real,TX<:Real}
    @inbounds for i in 1:f.n, j in 1:f.n
        P[i, j] = f.b(x[i] - j)
    end
    return P
end

function (f::PermutationVectorIHPenalized)(x::AbstractVector{TX}) where {TX<:Real}
    # Calculate permutation matrix
    create_perm_matrix!(f.P_buffer, f, x)

    # Compute A*P*A*P'
    mul!(f.AP_buffer, f.A, f.P_buffer)
    mul!(f.APA_buffer, f.AP_buffer, f.A)
    mul!(f.AP_buffer, f.APA_buffer, f.P_buffer')

    # Calculate penalty term
    penalty = tr(f.c_mat * f.P_buffer)

    # Return final result
    return -(tr(f.AP_buffer) - penalty)
end

function jacobian!(
    grad::AbstractVector{TG},
    f::PermutationVectorIHPenalized{T},
    x::AbstractVector{TX}
) where {T<:Real,TG<:Real,TX<:Real}
    # Calculate permutation matrix
    create_perm_matrix!(f.P_buffer, f, x)

    # Compute -2*A*P*A + diag(c)
    mul!(f.AP_buffer, f.A, f.P_buffer)
    mul!(f.APA_buffer, f.AP_buffer, f.A)
    rmul!(f.APA_buffer, -2)

    @inbounds for i in 1:f.n
        f.APA_buffer[i, i] += f.c_mat[i, i]
    end

    # Compute gradient
    fill!(grad, zero(TG))
    @inbounds for i in 1:f.n
        for j in 1:f.n
            grad[i] += f.APA_buffer[i, j] * f.db(x[i] - j)
        end
    end

    return grad
end

function jacobian(f::PermutationVectorIHPenalized{T}, x::AbstractVector{TX}) where {T<:Real,TX<:Real}
    grad = Vector{promote_type(T, eltype(x))}(undef, length(x))
    jacobian!(grad, f, x)
    return grad
end

function hessian!(
    H::AbstractMatrix{TH},
    f::PermutationVectorIHPenalized{T},
    x::AbstractVector{TX}
) where {T<:Real,TH<:Real,TX<:Real}
    # Calculate permutation matrix
    create_perm_matrix!(f.P_buffer, f, x)

    # Compute -2*A*P*A + diag(c)
    mul!(f.AP_buffer, f.A, f.P_buffer)
    mul!(f.APA_buffer, f.AP_buffer, f.A)

    @inbounds for i in 1:f.n
        f.APA_buffer[i, i] += f.c_mat[i, i]
    end

    # Pre-compute derivatives to avoid repeated function calls
    db_values = Matrix{T}(undef, f.n, f.n)
    d2b_values = Matrix{T}(undef, f.n, f.n)

    @inbounds for i in 1:f.n, j in 1:f.n
        db_values[i, j] = f.db(x[i] - j)
        d2b_values[i, j] = f.d2b(x[i] - j)
    end

    # Compute Hessian
    @inbounds for i in 1:f.n
        for j in 1:f.n
            # First term: uses pre-computed outer product
            term1 = zero(T)
            for k in 1:f.n, l in 1:f.n
                term1 -= 2 * f.A[i, j] * f.A[k, l] * db_values[i, l] * db_values[j, k]
            end

            # Second term: only applies when i == j
            term2 = zero(T)
            if i == j
                for l in 1:f.n
                    term2 -= 2 * f.APA_buffer[i, l] * d2b_values[i, l]
                end
            end

            H[i, j] = term1 + term2
        end
    end

    return H
end

function hessian(f::PermutationVectorIHPenalized{T}, x::AbstractVector{TX}) where {T<:Real,TX<:Real}
    H = Matrix{promote_type(T, eltype(x))}(undef, f.n, f.n)
    hessian!(H, f, x)
    return H
end

"""
    solve_with_custom_gradients(obj_func, max_iter=100000, tol=1e-8)

Solve an optimization problem using custom gradients for the Irwin-Hall permutation approximation.
"""
function solve_with_custom_gradients(
    obj_func::PermutationVectorIHPenalized,
    max_iter::Integer=100000,
    tol::Real=1e-8
)
    n = obj_func.n
    T = eltype(obj_func.A)
    x0 = binomial(n, 2) / n * ones(n)

    function cons!(res, x, p)
        @inbounds for j in 1:n
            res[j] = sum(obj_func.b(x[i] - j) for i in 1:n) - one(T)
        end
        return nothing
    end

    function cons_j!(J, x, p)
        @inbounds for j in 1:n, i in 1:n
            J[j, i] = obj_func.db(x[i] - j)
        end
        return nothing
    end

    function cons_h!(H, x, p)
        @inbounds for j in 1:n, i in 1:n
            H[j][i, i] = obj_func.d2b(x[i] - j)
        end
        return nothing
    end

    function grad!(G, x, p)
        jacobian!(G, obj_func, x)
        return nothing
    end

    function hess!(H, x, p)
        hessian!(H, obj_func, x)
        return nothing
    end

    optf = OptimizationFunction(
        (u, p) -> obj_func(u);
        grad=grad!,
        hess=hess!,
        cons=cons!,
        cons_j=cons_j!,
        cons_h=cons_h!
    )

    # Lower and upper bounds
    lb = zeros(T, n)
    ub = fill(T(n + 1), n)

    # Create the optimization problem
    prob = OptimizationProblem(
        optf,
        x0,
        nothing;
        lb=lb,
        ub=ub,
        lcons=zeros(T, n),
        ucons=zeros(T, n)
    )

    # Set up IPOPT solver options
    optimizer = OptimizationMOI.MOI.OptimizerWithAttributes(
        Ipopt.Optimizer,
        "tol" => Float64(tol),
        "max_iter" => max_iter,
        "hessian_approximation" => "limited-memory",
        "print_level" => 3,
        "max_cpu_time" => 3600.0  # 1 hour
    )

    # Solve the problem
    tick = time()
    sol = SciMLBase.solve(prob, optimizer)
    toc = time()
    solution_time = toc - tick

    return sol, x0, solution_time
end

"""
    IHMethod <: AbstractMethod

A method for computing approximate graph symmetries using Irwin-Hall distribution.

# Fields
- `name::String`: Name of the method
- `version::String`: Version identifier
- `parameters::Dict{Symbol, Any}`: Additional parameters
- `supported_metrics::Vector{Symbol}`: Metrics that this method can compute
"""
struct IHMethod <: AbstractMethod
    name::String
    version::String
    parameters::Dict{Symbol,Any}
end

"""
    IHMethod(;name="IH", version="v1", penalty=0.2, max_iter=100000, tol=1e-8)

Construct an IHMethod with the given parameters.
"""
function IHMethod(;
    name::String="IH",
    version::String="v1",
    penalty::Number=0.2,
    max_iter::Integer=100000,
    tol::Real=1e-8
)
    parameters = Dict{Symbol,Any}(
        :penalty => penalty,
        :max_iter => max_iter,
        :tol => tol
    )
    return IHMethod(name, version, parameters)
end

"""
    set_parameter!(method::IHMethod, key::Symbol, value)

Set a parameter for the method.
"""
function set_parameter!(method::IHMethod, key::Symbol, value)
    method.parameters[key] = value
    return method
end

"""
    get_parameter(method::IHMethod, key::Symbol, default=nothing)

Get a parameter from the method with an optional default value.
"""
function get_parameter(method::IHMethod, key::Symbol, default=nothing)
    get(method.parameters, key, default)
end

"""
    supported_metrics(method::IHMethod)

Get all metrics supported by this method.
"""
supported_metrics(::IHMethod) = [:time, :s_metric]

"""
    solve(method::IHMethod, instance::MatrixInstance)

Apply the Irwin-Hall method to a matrix instance.
Returns a Solution object containing the result and computed metrics.
"""
function solve(method::IHMethod, instance::AbstractInstance)
    # Get parameters
    penalty = get_parameter(method, :penalty, 0.2)
    max_iter = get_parameter(method, :max_iter, 100000)
    tol = get_parameter(method, :tol, 1e-8)

    # Create the objective function
    A = Instances.adjacency(instance)
    n = size(A, 1)
    penalty_vector = fill(convert(eltype(A), penalty), n)

    obj_func = PermutationVectorIHPenalized(A, penalty_vector)

    original_stdout = stdout
    rd, = redirect_stdout()
    sol, x0, execution_time = solve_with_custom_gradients(obj_func, max_iter, tol)
    redirect_stdout(original_stdout)
    closewrite(rd)
    solve_stdout = String(readavailable(rd))
    close(rd)

    # Convert solution to permutation
    perm = sortperm(sol.u) |> invperm
    P = to_permutation_matrix(perm)
    P_matrix = Matrix{Float64}(P)

    # Create solution object and add metrics
    solution = Solution(P_matrix, instance)

    set_metric!(solution, :time, execution_time)
    # set_metric!(solution, :s_metric, s_value)
    set_metric!(solution, :initial_point, x0)
    set_metric!(solution, :stdout, solve_stdout)

    return solution
end

export IHMethod, solve
