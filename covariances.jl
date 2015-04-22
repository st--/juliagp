abstract ScalarKernel <: Kernel
abstract VectorKernel <: Kernel

immutable CovConst{N<:Number} <: Kernel
    sf2::N
end
length(k::CovConst) = 1

*(sf2::Number, a::Kernel) = CovConst(sf2)*a
*(a::Kernel, sf2::Number) = a*CovConst(sf2)

cov(k::CovConst, x1, x2) = k.sf2
covderiv(k::CovConst, i::Integer, x1, x2) = one(k.sf2)


immutable CovSEiso{N<:Number} <: VectorKernel
    ell::N
end
typealias SqExp CovSEiso
length(k::CovSEiso) = 1

cov{T}(k::CovSEiso, x1::T, x2::T) = exp(-0.5vecnorm(x1 - x2)^2 / k.ell^2)

covderiv{T}(k::CovSEiso, i::Integer, x1::T, x2::T) = vecnorm(x1 - x2)^2 / k.ell^3 * cov(k, x1, x2)
# derivative w.r.t. i'th hyperparameter of covariance kernel

function cov_d{T}(k::CovSEiso, x1::T, x2::T)
# derivative w.r.t. x2
    (x1 - x2)/k.ell^2 * cov(k, x1, x2)
end

covd_(k::CovSEiso, x1, x2) = - cov_d(k, x1, x2) # derivative w.r.t. x1

function covdd{T}(k::CovSEiso, x1::T, x2::T) 
# second (mixed) derivative w.r.t x1 and x2
    cov(k, x1, x2) / k.ell^2 * (1 - (x1 - x2)^2/k.ell^2)
end

covd2(k::CovSEiso, x1, x2) = - covdd(k, x1, x2) # second derivative w.r.t. x1 or x2

immutable CovSEard{N<:Number} <: VectorKernel
    ells::Vector{N}
end
length(k::CovSEard) = length(k.ells)

cov{T<:Number}(k::CovSEard, x1::Vector{T}, x2::Vector{T}) = exp(-0.5vecnorm((x1 - x2) ./ k.ells)^2)

covderiv{T<:Number}(k::CovSEard, i::Integer, x1::Vector{T}, x2::Vector{T}) = (
    abs(x1 - x2)[i]^2 / k.ells[i]^3 * cov(k, x1, x2)
)


immutable CovPeriodic <: ScalarKernel
    ell::Number
    p::Number
end
length(k::CovPeriodic) = 2

cov(k::CovPeriodic, x1::Number, x2::Number) = exp(-2.sin(π*(x1-x2)/k.p)^2 / k.ell^2)

function covderiv(k::CovPeriodic, i::Integer, x1::Number, x2::Number)
    if i == 1 # ∂/∂(ell)
        return 4.sin(π*(x1-x2)/k.p)^2 / k.ell^3 * cov(k, x1, x2)
    elseif i == 2 # ∂/∂p
        arg = π*(x1-x2)/k.p
        return 4.sin(arg) / k.ell^2 * cos(arg) * arg / k.p * cov(k, x1, x2)
    end
end

