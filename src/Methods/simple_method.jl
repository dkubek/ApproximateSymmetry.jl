struct SimpleMethod <: AbstractMethod
    name::String
    version::String
    parameters::MethodParameters
end

SimpleMethod(name::String, version::String) = SimpleMethod(name, version, MethodParameters())

function supported_metrics(::SimpleMethod)
    return Symbol[:time, :s_metric]
end

function solve(::SimpleMethod, instance::AbstractInstance)
    n = size(Instances.adjacency(instance), 1)

    start_time = time()
    P = Matrix{Float64}(I, n, n)
    solve_time = time() - start_time

    solution = Solution(P, instance)
    set_metric!(solution, :time, solve_time)

    return solution
end
