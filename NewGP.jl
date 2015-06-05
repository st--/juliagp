module NewGP

include("MLKernels.jl/src/MLKernels.jl")
using MLKernels

import Base.mean, Base.std
export GPInput, GPOutput, TrainedGP, GPPrediction 

const jitterdefault = 1e-6

typealias GPInput{T<:FloatingPoint} Matrix{T}
typealias GPOutput{T<:FloatingPoint} Vector{T}

# for the implementation with Cholesky decomposition
immutable TrainedGP{N<:FloatingPoint}
    kernel::Kernel{N}
    noise::N
    xs::GPInput{N} # training input points
    ys::GPOutput{N} # training function values
    L::Matrix{N} # Cholesky factor of (training, training) covariance matrix
    alpha::Vector{N} # coefficients

    function TrainedGP(k::Kernel{N}, xs::GPInput{N}, ys::GPOutput{N}, noise::N, jitter::N)
        kxx = kernelmatrix(k, xs, xs)
        C = kxx + noise^2*I
        L = chol(C + jitter*I, :L)
        α = L' \ (L \ ys)
        new(k, noise, xs, ys, full(L), α)
    end
end
function traingp{N<:FloatingPoint}(k::Kernel{N}, xs::GPInput{N}, ys::GPOutput{N}, noise::N = zero(N), jitter::N = convert(N, jitterdefault))
    TrainedGP{N}(k, xs, ys, noise, jitter)
end

type GPPrediction{N<:FloatingPoint}
    gp::TrainedGP{N}
    newxs::GPInput{N}
    kxx_::Matrix{N}
    kx_x_::Matrix{N}
    has_mean::Bool
    has_std::Bool
    mean::GPOutput{N}
    std::GPOutput{N}

    function GPPrediction(gp::TrainedGP, newxs::GPInput)
        kxx_ = kernelmatrix(gp.kernel, gp.xs, newxs)
        kx_x_ = kernelmatrix(gp.kernel, newxs, newxs)
        new(gp, newxs, kxx_, kx_x_, false, false, N[], N[])
    end
end

function calcmean!{N<:FloatingPoint}(pred::GPPrediction{N})
    kx_x = pred.kxx_'
    pred.has_mean = true
    pred.mean = kx_x * pred.gp.alpha
end

function calcstd!{N<:FloatingPoint}(pred::GPPrediction{N})
    v = pred.gp.L \ pred.kxx_
    Σ = pred.kx_x_ - v'v
    pred.has_std = true
    pred.std = sqrt(diag(Σ))
end

mean{N<:FloatingPoint}(pred::GPPrediction{N}) = pred.has_mean ? pred.mean : calcmean!(pred)
std{N<:FloatingPoint}(pred::GPPrediction{N}) = pred.has_std ? pred.std : calcstd!(pred)

function logML(gp::TrainedGP)
    -0.5gp.ys' * gp.alpha - sum(log(diag(gp.L))) - 0.5log(2pi) * inputlength(gp.xs)
end

function logML_dp(gp::TrainedGP, derivs)
    n = length(derivs)
    lmldp = zeros(n)
    for i=1:n
        Cdp = kernelmatrix_dp(gp.kernel, gp.xs, derivs[i])
        lmldp[i] = 0.5gp.alpha' * Cdp * gp.alpha - 0.5trace(gp.kxxI * Cdp)
    end
    lmldp
end

end
