function gptrain(k::Kernel, xs::GPInput, ys::GPOutput, noise=0.; jitter=jitterdefault)
    kxx = covmat(k, xs, xs) + noise^2 * neye(xs) + jitter * neye(xs)
    kxxI = inv(kxx)
    α = kxxI * ys
    return GaussianProcess(k, noise, xs, ys, kxxI, α)
end

function predict(gp::GaussianProcess, newxs::GPInput)
    kxx_ = covmat(gp.kernel, gp.xs, newxs)
    kx_x = kxx_'
    kx_x_ = covmat(gp.kernel, newxs, newxs)
    
    μ = kx_x * gp.α
    Σ = kx_x_ - kx_x * gp.kxxI * kxx_
    
    return μ, sqrt(diag(Σ))
end

function covmatderiv(gp::GaussianProcess)
    nhyper = length(gp.kernel)
    J = zeros(eltype(gp.kxxI), size(gp.kxxI)..., nhyper+1)
    for m=1:nhyper
        for j=1:inputlength(gp.xs), i=1:inputlength(gp.xs)
            J[i,j,m] = covderiv(gp.kernel, m, inputindex(gp.xs, i), inputindex(gp.xs, j))
        end
    end
    # noise derivative:
    for i=1:inputlength(gp.xs)
        J[i,i,nhyper+1] = 2gp.noise
    end
    return J
end

logML(gp::GaussianProcess) = (
    -0.5 * gp.ys' * gp.kxxI * gp.ys
    + 0.5 * log(det(gp.kxxI)) # would be negative and log(det(kxx))
    - 0.5 * inputlength(gp.xs) * log(2π)
)

function logMLderiv(gp::GaussianProcess)
    kderiv = covmatderiv(gp)
    return Float64[(
        0.5 * gp.ys' * gp.kxxI * kderiv[:,:,i] * gp.kxxI * gp.ys
        - 0.5 * trace(gp.kxxI * (kderiv[:,:,i]))
        )[1] for i=1:length(gp.kernel)+1]
end
