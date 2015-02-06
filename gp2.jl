function gptrain2(k::Kernel, xs::GPInput, ys::GPOutput, noise=0.; jitter=jitterdefault)
    kxx = covmat(k, xs, xs)
    L = chol(kxx + noise^2 * neye(xs) + jitter * neye(xs), :L)
    α = L' \ (L \ ys)
    return GaussianProcess2(k, noise, xs, ys, full(L), α)
end

function predict(gp::GaussianProcess2, newxs::GPInput)
    kxx_ = covmat(gp.kernel, gp.xs, newxs)
    kx_x = kxx_'
    kx_x_ = covmat(gp.kernel, newxs, newxs)
    μ = kx_x * gp.α
    v = gp.L \ kxx_
    Σ = kx_x_ - v'*v
    return μ, sqrt(diag(Σ))
end

logML(gp::GaussianProcess2) = (
    -0.5 * gp.ys' * gp.α
    - sum(log(diag(gp.L)))
    - 0.5 * inputlength(gp.xs) * log(2π)
)

