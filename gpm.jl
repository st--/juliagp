# for implementation with mean function, but 1D only
abstract MeanFunction
# h = m x + n
type MeanLinear{N<:Number} <: MeanFunction
    m::N
    n::N
end
type MeanZero <: MeanFunction
end
type GaussianProcessM{N<:Number}
    kernel::Kernel
    mean::MeanFunction
    noise::N
    xs::Vector{N}
    ys::Vector{N}
    KyInv::Matrix{N}
    H::Matrix{N} # 1 x n matrix...
    HKyInv::Matrix{N}
    HKyInvHT_Inv::Matrix{N}
    beta::Matrix{N}
end


function meaneval{N<:Number}(m::MeanLinear{N}, xs::Vector{N})
    n = inputlength(xs)
    H = ones((2, n))
    H[2,:] = xs
    return H
end
#meaneval(m::MeanZero, xs::Vector) = zeros(xs)

function gptrainM{N<:Number}(k::Kernel, m::MeanFunction, xs::Vector{N}, ys::Vector{N}, noise=0.; jitter=jitterdefault)
    K = covmat(k, xs, xs)
    Ky = K + noise^2 * neye(xs)
    KyInv = inv(Ky)
    H = meaneval(m, xs)
    HKyInv = H * KyInv
    HKyInvHT_Inv = inv(HKyInv * H')
    beta = HKyInvHT_Inv * HKyInv * ys
    return GaussianProcessM(k, m, noise, xs, ys, KyInv, H, HKyInv, HKyInvHT_Inv, beta'')
end
function predict{N<:Number}(gp::GaussianProcessM{N}, newxs::Vector{N})
    K_ = covmat(gp.kernel, gp.xs, newxs)
    K__ = covmat(gp.kernel, newxs, newxs)
    H_ = meaneval(gp.mean, newxs)
    
    K_TKyInv = K_' * gp.KyInv
    R = H_ - gp.HKyInv * K_
    μ = (K_TKyInv * gp.ys) + R' * gp.beta
    
    #μ = H_' * gp.beta + K_' * gp.KyInv * (gp.ys - gp.H' * gp.beta)
    
    Σ = (K__ - K_TKyInv * K_) + R' * gp.HKyInvHT_Inv * R
    return μ, sqrt(diag(Σ))
end
function logML(gp::GaussianProcessM)
    A = gp.HKyInv * gp.H'
    C = gp.KyInv * gp.H' * inv(A) * gp.HKyInv
    n = inputlength(gp.xs)
    m = rank(gp.H')
    return (( -0.5 * gp.ys' * gp.KyInv * gp.ys) + ( 0.5 * gp.ys' * C * gp.ys)
    + ( 0.5 * logdet(gp.KyInv)) - ( 0.5 * logdet(A))
    - ( 0.5(n-m) * log(2π)))
end

