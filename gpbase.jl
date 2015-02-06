# Covariance matrix between input points
covmat{T}(kernel::Kernel, x1s::Vector{T}, x2s::Vector{T}) = T[
    cov(kernel, x1, x2) for x1 in x1s, x2 in x2s
]
covmat{T}(kernel::Kernel, x1s::Array{T,2}, x2s::Array{T,2}) = T[
    cov(kernel, vec(x1s[i,:]), vec(x2s[j,:])) for i=1:inputlength(x1s), j=1:inputlength(x2s)
]

function priorsample(k::Kernel, xs::GPInput; jitter=jitterdefault)
    kxx = covmat(k, xs, xs) + jitter * neye(xs)
    return rand(MvNormal(kxx))
end
function priorsample(k::Kernel, xs::GPInput, n; jitter=jitterdefault)
    kxx = covmat(k, xs, xs) + jitter * neye(xs)
    return rand(MvNormal(kxx), n)
end
