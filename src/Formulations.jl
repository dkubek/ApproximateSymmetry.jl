module Formulations

using ..Interfaces
using LinearAlgebra

# Exports
export AbstractFormulation, AbstractObjectiveFunction,
       get_objective_function, solve, objective_value,
       evaluate, gradient, gradient!, hessian, hessian!,
       WeightedAsymmetryDistance, AsymmetryOnly,
       frobenius_norm_asymmetry, frobenius_norm_asymmetry_gradient,
       frobenius_norm_asymmetry_hessian, distance_to_original,
       distance_to_original_gradient

# Include implementation files
include("formulations/abstract_formulation.jl")
include("formulations/common.jl")
include("formulations/objective_functions.jl")

end # module
