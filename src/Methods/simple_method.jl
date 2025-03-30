struct SimpleMethod <: AbstractMethod
    name::String
    version::String
    parameters::MethodParameters
end

SimpleMethod(name::String, version::String) = SimpleMethod(name, version, MethodParameters())

function supported_metrics(::SimpleMethod)
    return Symbol[:time, :s_metric]
end

function solve(::SimpleMethod, instance::Instances.MatrixInstance{T}) where {T<:Real}
    n = instance.n

    start_time = time()
    P = Matrix{T}(I, n, n)
    solve_time = time() - start_time
    
    solution = Solution(P, instance)
    set_metric!(solution, :time, solve_time)

    return solution
end