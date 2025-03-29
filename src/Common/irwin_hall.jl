"""
    irwin_hall_3(x)

Compute the probability density function of the Irwin-Hall distribution with n=3.
This is used to create a smoothed approximation of permutation matrices.
"""
function irwin_hall_3(x::T) where {T<:Real}
    if x < 0
        return zero(T)
    elseif x < 1
        return x^2 / 2
    elseif x < 2
        return -x^2 + 3x - 3/2
    elseif x < 3
        return (3 - x)^2 / 2
    else
        return zero(T)
    end
end

"""
    irwin_hall_3_derivative(x)

Compute the first derivative of the Irwin-Hall distribution with n=3.
"""
function irwin_hall_3_derivative(x::T) where {T<:Real}
    if x < 0
        return zero(T)
    elseif x < 1
        return x
    elseif x < 2
        return -2x + 3
    elseif x < 3
        return -(3 - x)
    else
        return zero(T)
    end
end

"""
    irwin_hall_3_hessian(x)

Compute the second derivative of the Irwin-Hall distribution with n=3.
"""
function irwin_hall_3_hessian(x::T) where {T<:Real}
    if x < 0
        return zero(T)
    elseif x < 1
        return one(T)
    elseif x < 2
        return -2*one(T)
    elseif x < 3
        return one(T)
    else
        return zero(T)
    end
end

"""
    periodic(f, period)

Create a periodic function with the given period.
"""
function periodic(f::Function, period::Integer)
    return x -> begin
        x_mod = mod(x, period)
        return f(x_mod)
    end
end