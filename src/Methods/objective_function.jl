"""
ObjectiveFunction is an abstract type representing mathematical objectives
with methods for computing values, gradients, and Hessians.
"""
abstract type ObjectiveFunction end

function (f::ObjectiveFunction)(x)
    throw(MethodError(f, (x,)))
end

function jacobian!(grad::AbstractVector, f::ObjectiveFunction, x::AbstractVector)
    throw(MethodError(jacobian!, (grad, f, x)))
end

function jacobian(f::ObjectiveFunction, x::AbstractVector)
    throw(MethodError(jacobian, (f, x)))
end

function hessian!(H::AbstractMatrix, f::ObjectiveFunction, x::AbstractVector)
    throw(MethodError(hessian!, (H, f, x)))
end

function hessian(f::ObjectiveFunction, x::AbstractVector)
    throw(MethodError(hessian, (f, x)))
end
